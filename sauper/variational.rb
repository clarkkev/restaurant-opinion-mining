#!/usr/bin/ruby
#
# Variational inference using ruby for food aspect of restaurants
#

require 'time'
require 'pp'
require 'rubygems'
require 'fastercsv'
require 'json'
require 'thread'
require 'facter'
require 'fileutils'
require 'logger'
require 'stemmer'

include Java

#    require 'jruby-prof'
#    
#    result = JRubyProf.profile do


STDERR.sync = true
srand 1  # for repeatable numbers
LOGTOLERANCE = 30.0
Facter.loadfacts
INIT_CONST = 5

# taylor approximation, taken from Blei c code
def digamma(x)
    x = x + 6
    p = 1.0 / (x * x)
    #p = x ** -2
    #p = (((0.004166666666667*p-0.003968253986254)*p+
    #    0.008333333333333)*p-0.083333333333333)*p
    #p = p + Math.log(x) - 0.5 / x - 1.0 / (x-1) - 1.0 / (x-2) -
    #    1.0 / (x-3) - 1.0 / (x-4) - 1.0 / (x-5) - 1.0 / (x-6)
    return (((0.004166666666667*p-0.003968253986254)*p+
        0.008333333333333)*p-0.083333333333333)*p + 
        Math.log(x) - 0.5 / x - 1.0 / (x-1) - 1.0 / (x-2) -
        1.0 / (x-3) - 1.0 / (x-4) - 1.0 / (x-5) - 1.0 / (x-6)
    #return p
end


class Hash
    def exp; Hash[*collect{|k,v| [k,Math.exp(v)]}.flatten]; end
    def round_to(x)
        Hash[*collect{|k,v| v.class == Float ? [k,v.round_to(x)] : [k,v] }.flatten]
    end
    def logtemp; $temp.nil? ? self : Hash[*collect{|k,v| [k,v*$temp]}.flatten]; end
    def lognorm; la = logadd; la == 0.0 ? self : Hash[*collect{|k,v| [k,v-la]}.flatten]; end
    def logadd
        max = max{|a,b| a[1] <=> b[1]}
        max_val = max[1]
        thresh = max_val - LOGTOLERANCE
        diff = select{|k,v| k != max[0] and v > thresh}.collect{|k,v| Math.exp(v-max_val)}.inject(:+)
        return diff.nil? ? max_val : max_val + Math.log(1.0 + diff)
    end
    def normalize; s = values.inject(:+); 
        if s != 1 then Hash[*collect{|k,v| [k,v/s]}.flatten] else self end
    end
end

class Array
    def normalize; s = inject(:+).to_f; if s != 1 then collect{ |x| x / s } else self end; end
    def ** (second); collect{|x| x**second}; end
    def temperature; $temp.nil? ? self : self**$temp; end
    def logtemp; $temp.nil? ? self : collect{|x| x * $temp}; end
    def sample
        r = rand
        index = length - 1 
        #temperature.normalize.each_with_index{ |x,i|
        normalize.each_with_index{ |x,i|
            if r < x then index = i; break; end 
            r = r - x
        } 
        index
    end
    def exp; collect{|x| Math.exp(x)}; end
    def log; collect{|x| Math.log(x)}; end
    def logsample; lognorm.exp.sample; end
    def lognorm; la = logadd; la == 0.0 ? self : collect{|x| x - la}; end
    def logadd
        max = each_with_index.max{|a,b| a[0] <=> b[0]}
        max_val = max[0]
        thresh = max_val - LOGTOLERANCE
        diff = each_with_index.select{|x| x[1] != max[1] and x[0] > thresh}.collect{|x| Math.exp(x[0]-max_val)}.inject(:+)
        return diff.nil? ? max_val : max_val + Math.log(1.0 + diff)
    end
    def split n
        count , fat_ones = self.size / n , self.size % n
        cp, out = self.dup, []
        out << cp.
            slice!(0, count + (out.size < fat_ones ? 1 : 0 )) while cp!=[]
        out
    end
    def truncate n
        if n == 0 
            return self
        else
            # probably inefficient
            #to_enum(:each_with_index).sort{|a,b| b[0] <=> a[0] }.to_enum(:each_with_index).collect{|e,i| i > n - 1 ? [0,e[1]] : e}.sort{|a,b| a[1] <=> b[1]}.collect{|e,i| e}

            # more efficient, i think.
            x = self
            cutoff = x.sort{|a,b| b <=> a}[n]
            collect{|e| e > cutoff ? e : 0}
        end
    end
    def round_to(x)
        collect{|i| i.class == Float ? i.round_to(x) : i }
    end
end

class Float
    def round_to(x)
        (self * 10**x).round.to_f / 10**x
    end
end

# constants

params = JSON.parse(open(ARGV[0]).read)

priors = [params["prior"]["P"],params["prior"]["A"],params["prior"]["B"],params["prior"]["I"]].normalize.log

$const = {
    #:restaurants => snippets.length,
    #:snippets => snippets.collect{|s| s.length},
    #:words => $word_counts.length,
    #:snippet_words => snippets.collect{|s| s.collect{|w| w.gsub(/(\w)(\W)/,"\\1 \\2").gsub(/(\W)(\w)/,"\\1 \\2").split.length}},
    #:snippet_words => snippets.collect{|s| s.collect{|w| w.split.length}},
    :clusters => params["length"]["properties"],
    :types => params["length"]["types"],
    :log_prior => {
        :P => priors[0], 
        :A => priors[1],
        :B => priors[2],
        :I => priors[3],
        :attributes => params["prior"]["attributes"].nil? ? 
                Math.log(1.0 / params["length"]["types"]) :
                params["prior"]["attributes"].normalize.log
        #:positive => Math.log(params["prior"]["positive"]),
        #:negative => Math.log(1 - params["prior"]["positive"]),
        #:flat => Math.log(1.0 / params["length"]["types"])
    },
    :seed => {
        :P => params["seed"]["P"], 
        :A => params["seed"]["A"], 
        :B => params["seed"]["B"],
        :I => params["seed"]["I"],
        :tau => params["seed"]["tau"]
    },
    :hmm => {
        :start => {
            :P => params["hmm"]["start"]["P"].nil? ? 0 : params["hmm"]["start"]["P"],
            :A => params["hmm"]["start"]["A"].nil? ? 0 : params["hmm"]["start"]["A"],
            :B => params["hmm"]["start"]["B"].nil? ? 0 : params["hmm"]["start"]["B"],
            :I => params["hmm"]["start"]["I"].nil? ? 0 : params["hmm"]["start"]["I"],
            :end => params["hmm"]["start"]["end"].nil? ? 0 : params["hmm"]["start"]["end"]
        },
        :P => {
            :P => params["hmm"]["P"]["P"].nil? ? 0 : params["hmm"]["P"]["P"],
            :A => params["hmm"]["P"]["A"].nil? ? 0 : params["hmm"]["P"]["A"],
            :B => params["hmm"]["P"]["B"].nil? ? 0 : params["hmm"]["P"]["B"],
            :I => params["hmm"]["P"]["I"].nil? ? 0 : params["hmm"]["P"]["I"],
            :end => params["hmm"]["P"]["end"].nil? ? 0 : params["hmm"]["P"]["end"]
        },
        :A => {
            :P => params["hmm"]["A"]["P"].nil? ? 0 : params["hmm"]["A"]["P"],
            :A => params["hmm"]["A"]["A"].nil? ? 0 : params["hmm"]["A"]["A"],
            :B => params["hmm"]["A"]["B"].nil? ? 0 : params["hmm"]["A"]["B"],
            :I => params["hmm"]["A"]["I"].nil? ? 0 : params["hmm"]["A"]["I"],
            :end => params["hmm"]["A"]["end"].nil? ? 0 : params["hmm"]["A"]["end"]
        },
        :B => {
            :P => params["hmm"]["B"]["P"].nil? ? 0 : params["hmm"]["B"]["P"],
            :A => params["hmm"]["B"]["A"].nil? ? 0 : params["hmm"]["B"]["A"],
            :B => params["hmm"]["B"]["B"].nil? ? 0 : params["hmm"]["B"]["B"],
            :I => params["hmm"]["B"]["I"].nil? ? 0 : params["hmm"]["B"]["I"],
            :end => params["hmm"]["B"]["end"].nil? ? 0 : params["hmm"]["B"]["end"]
        },
        :I => {
            :P => params["hmm"]["I"]["P"].nil? ? 0 : params["hmm"]["I"]["P"],
            :A => params["hmm"]["I"]["A"].nil? ? 0 : params["hmm"]["I"]["A"],
            :B => params["hmm"]["I"]["B"].nil? ? 0 : params["hmm"]["I"]["B"],
            :I => params["hmm"]["I"]["I"].nil? ? 0 : params["hmm"]["I"]["I"],
            :end => params["hmm"]["I"]["end"].nil? ? 0 : params["hmm"]["I"]["end"]
        }
    },
#    :sticky => {
#        :P => params["sticky"]["P"],
#        :A => params["sticky"]["A"],
#        :B => params["sticky"]["B"],
#        :start => {
#            :P => params["sticky"]["start"]["P"],
#            :A => params["sticky"]["start"]["A"],
#            :B => params["sticky"]["start"]["B"]
#        },
#        :end => {
#            :P => params["sticky"]["end"]["P"],
#            :A => params["sticky"]["end"]["A"],
#            :B => params["sticky"]["end"]["B"]
#        }
#    },
    :temp_step => params["iters"]["temp_step"],
    :temp_incr => params["iters"]["temp_incr"],
    :iters => params["iters"]["max_iters"],
    :output_freq => params["iters"]["output_freq"],
    :processors => params["iters"]["processors"].nil? ? 0 : params["iters"]["processors"],
    :lambda => {
        :P => params["lambda"]["P"],
        :A => params["lambda"]["A"],
        :B => params["lambda"]["B"],
        :psi => params["lambda"]["psi"],
        :phi => params["lambda"]["phi"],
        :tau => params["lambda"]["tau"],
        :eta => params["lambda"]["eta"].nil? ? 0.1 : params["lambda"]["eta"]
    },
    :hacks => {
        :pos => params["hacks"]["pos"].nil? ? false : params["hacks"]["pos"],
        :truncate => {
            :eta => params["hacks"]["truncate"]["eta"].nil? ? 0 : params["hacks"]["truncate"]["eta"],
            :P => params["hacks"]["truncate"]["P"].nil? ? 0 : params["hacks"]["truncate"]["P"],
            :A => params["hacks"]["truncate"]["A"].nil? ? 0 : params["hacks"]["truncate"]["A"]
        },
        :antibg => params["hacks"]["antibg"].nil? ? false : params["hacks"]["antibg"],
        :sep_props => params["hacks"]["sep_props"].nil? ? false : params["hacks"]["sep_props"],
        :use_trans => params["hacks"]["use_trans"].nil? ? false : params["hacks"]["use_trans"],
        :learn_trans => params["hacks"]["learn_trans"].nil? ? false : params["hacks"]["learn_trans"],
        :stem => params["hacks"]["stem"].nil? ? false : params["hacks"]["stem"],
        :spillover => params["hacks"]["spillover"].nil? ? 0.0 : params["hacks"]["spillover"],
        :numbers => params["hacks"]["numbers"].nil? ? false : params["hacks"]["numbers"]
    },
    :output_dir => params["run"]["output"],
    :phrase_file => params["run"]["phrases"],
    :seed_file => params["run"]["seed"]
}

$files = open($const[:phrase_file]).readlines.collect{|l| l.strip()}

if $const[:hacks][:pos]
    $snippets = $files.collect{|f| open(f).readlines.collect{|m| m.strip}.collect{|m| 
        m.split.collect{|w|
                tag = w.scan(/[^_]+$/)[0]
                w = w.sub(/_[^_]+$/,"")
                w.gsub(/([A-Za-z0-9.'-])([^A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([^A-Za-z0-9.'-])([A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([a-zA-Z'-])(\d)/,"\\1 \\2").gsub(/(\d)([A-Za-z'-])/,"\\1 \\2").split(/\s+/).collect{|wp| wp+"_"+tag}
        }.flatten.join(" ")
    }.select{|m| m.split.length > 2}}
else
    $snippets = $files.collect{|f| open(f).readlines.collect{|m| m.strip}.collect{|m| m.gsub(/([A-Za-z0-9.'-])([^A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([^A-Za-z0-9.'-])([A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([a-zA-Z'-])(\d)/,"\\1 \\2").gsub(/(\d)([A-Za-z'-])/,"\\1 \\2").gsub(/\s+/," ")}.select{|m| m.split.length > 2}}
end
    
#    m.gsub(/([A-Za-z0-9_. ](?=[^A-Za-z0-9_. ])|[^A-Za-z0-9_. ](?=[A-Za-z0-9_. ])|[a-zA-Z_ ](?=[^A-Za-z_ ])|[^A-Za-z_ ](?=[A-Za-z_ ]))/,"\\1 ")}.select{|m| m.split.length > 2}}

#$words = $snippets.collect{|r| r.collect{|s| s.split.collect{|w| w.downcase }}}
$words = $snippets.collect{|r| r.collect{|s|
    if $const[:hacks][:pos]
        s = s.gsub(/_[^_]+(\s|$)/,"\\1")
    end
   s.gsub(/([A-Za-z0-9.'-])([^A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([^A-Za-z0-9.'-])([A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([a-zA-Z'-])(\d)/,"\\1 \\2").gsub(/(\d)([A-Za-z'-])/,"\\1 \\2").split(/\s+/).collect{|w| 
            $const[:hacks][:stem] ? w.downcase.stem : w.downcase }.collect{|w|
            if $const[:hacks][:numbers] and w =~ /^\d+$/
               "#INTEGER"
            elsif $const[:hacks][:numbers] and w =~ /^\d*\.\d+$/
               "#FLOAT"
            else
               w 
            end }}}

if $const[:hacks][:pos]
    $pos = $snippets.collect{|r| r.collect{|s|
        s.split.collect{|w| 
            tag = w.scan(/[^_]+$/)[0]
            w = w.sub(/_[^_]+$/,"")
            w.gsub(/([A-Za-z0-9.'-])([^A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([^A-Za-z0-9.'-])([A-Za-z0-9.'-])/,"\\1 \\2").gsub(/([a-zA-Z'-])(\d)/,"\\1 \\2").gsub(/(\d)([A-Za-z'-])/,"\\1 \\2").split(/\s+/).collect{|wp| tag}}.flatten
    } }
else
    $pos = $words.collect{|r| r.collect{|s| s.collect{|w| ""} } }
end

#$word_counts = snippets.flatten.collect{|s| s.gsub(/(\w)(\W)/,"\\1 \\2").gsub(/(\W)(\w)/,"\\1 \\2").split(/\s/).collect{|w| w.downcase}}.flatten.inject(Hash.new(0)) {|h,x| h[x]+=1;h}

$word_counts = $words.flatten.inject(Hash.new(0)) {|h,x| h[x]+=1;h}
$w_index = Hash[*$word_counts.to_a.to_enum(:each_with_index).collect{|e,i| [e[0],i]}.flatten]
$w_name = $w_index.invert

$pos_counts = $pos.flatten.inject(Hash.new(0)) {|h,x| h[x]+=1;h}
$pos_index = Hash[*$pos_counts.to_a.to_enum(:each_with_index).collect{|e,i| [e[0],i]}.flatten]
$pos_name = $pos_index.invert

$const[:restaurants] = $snippets.length
$const[:snippets] = $snippets.collect{|s| s.length}
$const[:words] = $word_counts.length
$const[:pos] = $pos_counts.length
$const[:snippet_words] = $snippets.collect{|s| s.collect{|w| w.split.length}}

seed_list = Array.new($const[:types]){Hash.new(0)}
if $const[:hacks][:sep_props]
    props_seed = Array.new($const[:restaurants]){Hash.new(0)}
else
    props_seed = Hash.new(0)
end
if $const[:seed_file] != "none"
    open($const[:seed_file]).readlines.collect{|l| l.strip}.each{|w| 
        w = w.split
        word = $const[:hacks][:stem] ? w[0].stem : w[0]
        case w[1]
            when "TYPE"
                if w[2]
                    seed_list[w[2].to_i][word] = 1
                else
                    seed_list.each_with_index{|e,i|
                        seed_list[i][word] = 1
                    }
                end
            when "PROPERTIES" 
                if $const[:hacks][:sep_props]
                    if w[2]
                        props_seed[w[2].to_i][word] = 1
                    else
                        props_seed.each_with_index{|e,i|
                            props_seed[i][word] = 1
                        }
                    end
                else
                    props_seed[word] = 1
                end
        end
    }
end

procs = Facter["processorcount"].value.to_i

if $const[:processors] > 0
    procs = $const[:processors]
end

if $const[:restaurants] < procs 
    procs = $const[:restaurants]
end

PROCS = procs
#PROCS = 1



out = nil

if $const[:output_dir] != "none"
    if not File.exist?($const[:output_dir])
        FileUtils.makedirs($const[:output_dir])
    end

    dirs = Dir.glob("#{$const[:output_dir]}/out.*")
    num = dirs.length
    while (out.nil?)
        newdir = "#{$const[:output_dir]}/out.#{num}"
        if not File.exist?(newdir)
            FileUtils.mkdir(newdir)
            out = newdir
            $stderr.puts "Outputting to #{out}."
        end
        num += 1
    end
end

if out 
    open("#{out}/params", "w"){|f|
        f.syswrite(params.to_json + "\n")
    }
    log = Logger.new("#{out}/log")
else
    log = Logger.new(STDERR)
end

log.datetime_format = "%Y-%m-%d %H:%M:%S.%L "

# arrays

class Values
    attr_accessor :values
    def initialize()
    end
end

class Zw < Values
    def initialize()
        super()
        # current distribution 
        @values = Array.new($const[:restaurants]).to_enum(:each_with_index).collect{|e,i| 
            Array.new($const[:snippets][i]).to_enum(:each_with_index).collect{|f,j| 
                Array.new($const[:snippet_words][i][j]).to_enum(:each_with_index).collect{|g,k|
                    {:A => nil, :P => nil, :B => nil, :I => nil}
                }
            }
        }
    end
    def update(rest, snip, n, m, word)
        changes = @values[rest][snip][word]
        if ($const[:hacks][:sep_props])
            $properties.values[rest][snip].each_with_index{|pp,p|
                change = pp * changes[:P]
                $theta.P.update(rest,p,n,change)
            }
        else
            $properties.values[rest][snip].each_with_index{|pp,p|
                change = pp * changes[:P]
                $theta.P.update(p,n,change)
            }
        end
        $attitudes.values[rest][snip].each_with_index{|pa,a|
            change = pa * changes[:A]
            $theta.A.update(a,n,change)
        }
        $theta.B.update(n,changes[:B])
        $theta.I.update(n,changes[:I])

        if $const[:hacks][:pos] 
            $eta.P.update(m,changes[:P])
            $eta.A.update(m,changes[:A])
            $eta.B.update(m,changes[:B])
            $eta.I.update(m,changes[:I])
        end

        #sticky values for HMM
        if word > 0
            $Zw.values[rest][snip][word-1].each{|z1,pz1|
                changes.each{|z2,pz2| @tau_count[z1][z2] += pz1 * pz2 } 
            }
        else
            changes.each{|z2,pz2| @tau_count[:start][z2] += pz2 } 
        end

        if word < $const[:snippet_words][rest][snip] - 1
            $Zw.values[rest][snip][word+1].each{|z2,pz2|
                changes.each{|z1,pz1| @tau_count[z1][z2] += pz1 * pz2 } 
            }
        else
            changes.each{|z1,pz1| @tau_count[z1][:end] += pz1 } 
        end 
    end
    def sample(rest, snip, n, m, word)
        update = {}

        update[:P] = $const[:log_prior][:P]
        update[:A] = $const[:log_prior][:A]
        update[:B] = $const[:log_prior][:B]
        update[:I] = $const[:log_prior][:I]

        # for all properties and attitudes, sample Zw's
        #if ($const[:hacks][:sep_props])
        #    update[:P] += $properties.values[rest][snip].to_enum(:each_with_index).inject(:+){|pp,p| p[0] * $theta.P.logprob[rest][p[1]][n]}
        #else
        #    update[:P] += $properties.values[rest][snip].to_enum(:each_with_index).inject(:+){|pp,p| p[0] * $theta.P.logprob[p[1]][n]}
        #end
        if ($const[:hacks][:sep_props])
            $properties.values[rest][snip].each_with_index{|pp,p|
                    update[:P] += pp * $theta.P.logprob[rest][p][n]
            }
        else
            $properties.values[rest][snip].each_with_index{|pp,p|
                    update[:P] += pp * $theta.P.logprob[p][n]
            }
        end
        #update[:A] += $attitudes.values[rest][snip].to_enum(:each_with_index).inject(:+){|pa,a| a[0] * $theta.A.logprob[a[1]][n]}
        $attitudes.values[rest][snip].each_with_index{|pa,a|
            update[:A] += pa * $theta.A.logprob[a][n]
        }
        update[:B] += $theta.B.logprob[n]
        update[:I] += $theta.I.logprob[n]

        # pos values
        if $const[:hacks][:pos]
            update[:P] += $eta.P.logprob[m]
            update[:A] += $eta.A.logprob[m]
            update[:B] += $eta.B.logprob[m]
            update[:I] += $eta.I.logprob[m]
        end

        if $const[:hacks][:use_trans]
            #sticky values for HMM
            if word > 0
                $Zw.values[rest][snip][word-1].each{|z,pz|
                    if not pz.nil?
                        update.each{|y,py| update[y] += pz * $tau.logprob[z][y] } 
                    end
                }
            else
                update.each{|y,py| update[y] += $tau.logprob[:start][y]}
            end
            if word < $const[:snippet_words][rest][snip] - 1
                $Zw.values[rest][snip][word+1].each{|z,pz|
                    if not pz.nil?
                        update.each{|y,py| update[y] += pz * $tau.logprob[y][z] } 
                    end
                }
            else
                update.each{|y,py| update[y] += $tau.logprob[y][:end]}
            end 
        end

        @values[rest][snip][word] = update.lognorm.exp
        #@values[rest][snip][word] = update.logtemp.lognorm.exp
    end
end

class Properties < Values
    def initialize()
        super()
        # current distribution
        @values = Array.new($const[:restaurants]).to_enum(:each_with_index).collect{|e,i| 
            Array.new($const[:snippets][i]){|f,j|
                Array.new($const[:clusters],nil)
            }
        }
    end
    def update(rest,snip) 
        @values[rest][snip].each_with_index{|change,p|
            $psi.update(rest,p,change)
        }
    end
    def sample(rest, snip)
        update = Array.new($const[:clusters])
        update.each_with_index{|x,p| 

            # distribution over properties
            prob = $psi.logprob[rest][p]

            #word_prob = 0

            # distributions over words
            if not $Zw.values[rest][snip][0][:P].nil?
                if ($const[:hacks][:sep_props])
                    $Zw.values[rest][snip].each_with_index{|z,j|
                        word = $words[rest][snip][j]
                        n = $w_index[word]

                        # distribution over property words
                        prob += z[:P] * $theta.P.logprob[rest][p][n]
                    }
                else
                    $Zw.values[rest][snip].each_with_index{|z,j|
                        word = $words[rest][snip][j]
                        n = $w_index[word]

                        # distribution over property words
                        prob += z[:P] * $theta.P.logprob[p][n]
                    }
                end
            end

            #prob += word_prob / $Zw.values[rest][snip].length

            # distribution over attitudes            
            if not $attitudes.values[rest][snip][0].nil?

                $attitudes.values[rest][snip].each_with_index{|pa,a|
                    prob += pa * $phi.logprob[rest][p][a]
                }

            end

            update[p] = prob
        }
        @values[rest][snip] = update.lognorm.exp
        #@values[rest][snip] = update.logtemp.lognorm.exp
    end
end

class Attitudes < Values
    def initialize()
        super()
        @values = Array.new($const[:restaurants]).to_enum(:each_with_index).collect{|e,i| 
            Array.new($const[:snippets][i]){|f,j|
                Array.new($const[:types],nil)
            }
        }
    end
    def update(rest,snip)
        @values[rest][snip].each_with_index{|change,a|
            $properties.values[rest][snip].each_with_index{|pp,p| 
                c = pp * change
                $phi.update(rest, p, a, c)
            }
        }
    end
    def sample(rest,snip)
        ## prior?
        #update = { :positive => $const[:log_prior][:positive],
        #           :negative => $const[:log_prior][:negative] }
        update = Array.new($const[:log_prior][:attributes])

        $properties.values[rest][snip].each_with_index{|pp,p| 
            update = update.to_enum(:each_with_index).collect{|value,j|
                value + pp * $phi.logprob[rest][p][j]
            }
            #update[:negative] += pp * $phi.logprob[rest][p][:negative]
            #update[:positive] += pp * $phi.logprob[rest][p][:positive]
        }
        if not $Zw.values[rest][snip][0][:A].nil?
        $Zw.values[rest][snip].each_with_index{|z,i|
            #if not z[:A].nil?
                word = $words[rest][snip][i]
                n = $w_index[word]

                update = update.to_enum(:each_with_index).collect{|value,j|
                    value + z[:A] * $theta.A.logprob[j][n] 
                }
                #update[:negative] += z[:A] * $theta.A.logprob[:negative][n]
                #update[:positive] += z[:A] * $theta.A.logprob[:positive][n] 
            #end
        }
        end

        @values[rest][snip] = update.lognorm.exp
        #@values[rest][snip] = update.logtemp.lognorm.exp
    end
end


# counters

class Counter
    attr_accessor :lambda,:count,:logprob,:mutex
    def initialize(l)
        @lambda = l.to_f
        @mutex = Mutex.new
    end
end

class Psi < Counter
    def initialize(l)
        super(l)
        @count = Array.new($const[:restaurants]){
            Array.new($const[:clusters], 0) }
        @logprob = Array.new($const[:restaurants]){
            Array.new($const[:clusters], 0) }
    end
    def update(rest, p, change)
            @count[rest][p] += change
    end
    def clear
        @count = Array.new($const[:restaurants]){
            Array.new($const[:clusters], 0) }
    end
    def renorm
        @logprob = @count.collect{|rest|
            rest.collect{|clust|
                digamma(clust + @lambda)
            }.lognorm 
        }
        clear()
    end
    def rand_init
        @logprob = @logprob.collect{|r| 
            r.collect{|c| Math.log(@lambda + INIT_CONST * rand)}.lognorm }
    end
end

# Rest level distribution over attitude for each property
class Phi < Counter
    def initialize(l)
        super(l)
        @count = Array.new($const[:restaurants]){
            Array.new($const[:clusters]){
                Array.new($const[:types],0)
                #{ :positive => 0, :negative => 0}
            }}
        @logprob = Array.new($const[:restaurants]){
            Array.new($const[:clusters]){
                Array.new($const[:types],0)
                #{ :positive => 0, :negative => 0}
            }}
    end
    def update(rest, p, a, change)
        @count[rest][p][a] += change
    end
    def clear
        @count = Array.new($const[:restaurants]){
            Array.new($const[:clusters]){
                Array.new($const[:types],0)
                #{ :positive => 0, :negative => 0}
            }}
    end
    def renorm
        @logprob = @count.collect{|rest|
            rest.collect{|clust|
                clust.collect{|value|
                    digamma(value + @lambda)
                }.lognorm
            }
        }
        clear()
    end
    def rand_init
        @logprob = @logprob.collect{|r| r.collect{|c| 
            Array.new($const[:types]){@lambda + INIT_CONST * rand}.lognorm
            #a = [Math.log(@lambda + INIT_CONST * rand), 
            #     Math.log(@lambda + INIT_CONST * rand)].lognorm;
            #{:positive => a[0], :negative => a[1]}
        }}
    end
end

class Tau < Counter
    attr_accessor :seed, :scale
    def initialize(l, seed, scale)
        super(l)
        @seed = seed
        @scale = scale 
        @count = {
            :start => {:P => 0, :A => 0, :B => 0, :I => 0},
            :P => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :A => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :B => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :I => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
        }
        @logprob = {
            :start => {:P => Math.log(0.3), :A => Math.log(0.1), :B => Math.log(0.6), :I => Math.log(0)}.lognorm,
            :P => {:P => Math.log(0.5), :A => Math.log(0.25), :B => Math.log(0.25), :I => Math.log(0), :end => Math.log(0.2)}.lognorm,
            :A => {:P => Math.log(0.3), :A => Math.log(0.4), :B => Math.log(0.3), :I => Math.log(0), :end => Math.log(0.55)}.lognorm,
            :B => {:P => Math.log(0.3), :A => Math.log(0.3), :B => Math.log(0.4), :I => Math.log(0), :end => Math.log(0.25)}.lognorm,
            :I => {:P => Math.log(0), :A => Math.log(0), :B => Math.log(0), :I => Math.log(0.9), :end => Math.log(0.1)}.lognorm
        }
    end
    def update(z1,z2,change)
        @count[z1][z2] += change
    end
    def clear
        @count = {
            :start => {:P => 0, :A => 0, :B => 0, :I => 0},
            :P => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :A => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :B => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :I => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
        }
    end
    def renorm
        if $const[:hacks][:learn_trans]
            @count.each{|z1,counts|
                digammas = {}
                counts.each{|z2,value|
                    #boost = @lambda
                    #if z1 == z2
                    #    boost += @scale * @seed[z1]
                    #elsif z1 == :start
                    #    boost += @scale * @seed[:start][z2]
                    #elsif z2 == :end
                    #    boost += @scale * @seed[:end][z1]
                    #end
                    #digammas[z2] = digamma(value + boost)
                    digammas[z2] = digamma(value + @lambda + @scale * @seed[z1][z2])
                }
                @logprob[z1] = digammas.lognorm
            }
        end
#        if $const[:hacks][:learn_trans]
#            @logprob = @count.collect{|z1,counts|
#                counts.collect{|z2,value|
#                    boost = @lambda
#                    if z1 == z2
#                        boost += @scale * @seed[z1]
#                    elsif z1 == :start
#                        boost += @scale * @seed[:start][z2]
#                    elsif z2 == :end
#                        boost += @scale * @seed[:end][z1]
#                    end
#                    digamma(value + boost)
#                }.lognorm
#            }
#        end
        clear()
    end
    def rand_init
        if $const[:hacks][:learn_trans]
            @logprob.keys.each{|z1|
                @logprob[z1].keys.each{|z2|
                    #boost = @lambda
                    #if z1 == z2
                    #    boost += @scale * @seed[z1]
                    #elsif z1 == :start
                    #    boost += @scale * @seed[:start][z2]
                    #elsif z2 == :end
                    #    boost += @scale * @seed[:end][z1]
                    #end
                    #@logprob[z1][z2] = Math.log(boost + INIT_CONST * rand)
                    @logprob[z1][z2] = Math.log(@lambda + @scale * @seed[z1][z2] + rand)#INIT_CONST * rand)
                }
                @logprob[z1] = @logprob[z1].lognorm
            }
        end
    end
end

class Rest_P < Counter
    attr_accessor :seed, :props
    def initialize(l, seed, props)
        super(l)
        @seed = seed
        @count = Array.new($const[:restaurants]){ Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) } }
        @logprob = Array.new($const[:restaurants]){ Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) } }
        @props = props
    end
    def clear
        @count = Array.new($const[:restaurants]){ Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) } }
    end
    def renorm
        @logprob = @count.to_enum(:each_with_index).collect{|rest,i|
            rest.collect{|clust|
                #clust.truncate($const[:hacks][:truncate][:P]).each_with_index{|word,k|
                clust.to_enum(:each_with_index).collect{|word,k|
                    digamma(word + @lambda + @seed * @props[i][$w_name[k]])
                }.lognorm
            }
        }
        clear()
    end
    def update(rest, p, n, change)
        @count[rest][p][n] += change
    end
    def update_spillover(counts)
        @mutex.synchronize{
            @count.each_with_index{|rest,i|
                rest.each_with_index{|clust,j|
                    clust.each_with_index{|word,k|
                        @count[i][j][k] += counts[k]
                    }
                }
            }
        }
    end
    def rand_init
        @logprob = @logprob.to_enum(:each_with_index).collect{|l,i| 
            l.collect{|r| r.to_enum(:each_with_index).collect{|m,j| 
                Math.log(@lambda + INIT_CONST * rand + 
                         @props[i][$w_name[j]] * @seed)
            }.lognorm } }
    end
end

class P < Counter
    attr_accessor :seed, :props
    def initialize(l, seed, props)
        super(l)
        @seed = seed
        @count = Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) }
        @logprob = Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) }
        @props = props
    end
    def clear
        @count = Array.new($const[:clusters]){ 
            Array.new($const[:words], 0) }
    end
    def renorm
        @logprob = @count.collect{|clust|
            #clust.truncate($const[:hacks][:truncate][:P]).each_with_index{|word,j|
            clust.to_enum(:each_with_index).collect{|word,j|
                #boost = @lambda + @seed * @props[$w_name[j]]
                digamma(word + @lambda + @seed * @props[$w_name[j]])
            }.lognorm
        }
        clear()
    end
    def update(p, n, change)
        @mutex.synchronize {
            @count[p][n] += change
        }
    end
    def update_spillover(counts)
        @mutex.synchronize{
            rest.each_with_index{|clust,j|
                clust.each_with_index{|word,k|
                    @count[j][k] += counts[k]
                }
            }
        }
    end
    def rand_init
        @logprob = @logprob.collect{|l| 
            l.to_enum(:each_with_index).collect{|m,j| 
                Math.log(@lambda + INIT_CONST * rand + @props[$w_name[j]] * @seed)
            }.lognorm 
        }
    end
end

class A < Counter
    attr_accessor :seed, :seed_list 
    def initialize(l, seed, seed_list)
        super(l)
        @seed = seed
        @count = Array.new($const[:types]){ 
            Array.new($const[:words], 0) }
        @logprob = Array.new($const[:types]){|j|
            Array.new($const[:words],0)}
            #{|i| Math.log(@lambda + @seed * seed_list[j][$w_name[i]])}.lognorm }
        @seed_list = seed_list
    end
    def clear
        @count = Array.new($const[:types]){ 
            Array.new($const[:words], 0) }
    end
    def renorm
        @logprob = @count.to_enum(:each_with_index).collect{|words,i|
            #words.truncate($const[:hacks][:truncate][:A]).each_with_index{|word,j|
            words.to_enum(:each_with_index).collect{|word,j|
                #seed_value = @seed_list[i][$w_name[j]] 
                #boost = @lambda + @seed * seed_value
                digamma(word + @lambda + @seed * @seed_list[i][$w_name[j]])
            }.lognorm
            #fail (digammas.inspect) if digammas.nil?
        }
        clear()
    end
    def update(a, n, change)
        @mutex.synchronize {
            @count[a][n] += change
        }
    end
    def rand_init
        @logprob = @logprob.to_enum(:each_with_index).collect{|l,i| 
            l.to_enum(:each_with_index).collect{|m,j| 
                Math.log(@lambda + INIT_CONST * rand + 
                         @seed_list[i][$w_name[j]] * @seed)
            }.lognorm 
        }
    end
end

class B < Counter
    attr_accessor :scale
    def initialize(l, scale,exempt)
        super(l)
        @scale = scale
        @count = Array.new($const[:words], 0)
        @logprob = Array.new($const[:words], 0)
        @exempt = exempt
    end
    #TODO: CHECK
    def clear
        @count = Array.new($const[:words]){|i| 
            @lambda + @scale * $word_counts[$w_name[i]]}
    end
    def renorm
        @logprob = @count.to_enum(:each_with_index).collect{|word,i|
            #w = $w_name[i]
            seed_value = @exempt[$w_name[i]] == 1 ? 0 : $word_counts[$w_name[i]]
            value = ($const[:hacks][:antibg] and @exempt[$w_name[i]] == 1) ? 0 : word
            #boost = @lambda + @scale * seed_value
            digamma(value + @lambda + @scale * seed_value)
        }.lognorm
        #fail (digammas.inspect) if digammas.nil?
        clear()
    end
    def update(n, change)
        @mutex.synchronize {
            @count[n] += change
        }
    end
    def rand_init
        @logprob = @logprob.to_enum(:each_with_index).collect{|l,i| 
            word = $w_name[i]
            boost = @exempt[word] == 1 ? 0 : @scale * $word_counts[word]
            Math.log(@lambda + INIT_CONST * rand + boost)
        }.lognorm
    end
end

class I < Counter
    attr_accessor :scale
    def initialize(l, scale)
        super(l)
        @scale = scale
        @count = Array.new($const[:words], 0)
        @logprob = Array.new($const[:words], 0)
    end
    #TODO: CHECK
    def clear
        @count = Array.new($const[:words]){|i| 
            @lambda + @scale * $word_counts[$w_name[i]]}
    end
    def renorm
        @logprob = @count.to_enum(:each_with_index).collect{|word,i|
            #w = $w_name[i]
            seed_value = $word_counts[$w_name[i]]
            value = word
            #boost = @lambda + @scale * seed_value
            digamma(value + @lambda + @scale * seed_value)
        }.lognorm
        #fail (digammas.inspect) if digammas.nil?
        clear()
    end
    def update(n, change)
        @mutex.synchronize {
            @count[n] += change
        }
    end
    def rand_init
        @logprob = @logprob.to_enum(:each_with_index).collect{|l,i| 
            word = $w_name[i]
            boost = @scale * $word_counts[word]
            Math.log(@lambda + INIT_CONST * rand + boost)
        }.lognorm
    end
end

class Theta
    attr_accessor :P, :A, :B, :I
    def initialize(l, aseed, seed_list, pseed, props, bscale, iscale)
        if $const[:hacks][:sep_props]
            @P = Rest_P.new(l[:P],pseed,props)
            @B = B.new(l[:B], bscale, seed_list.inject(:merge).merge(props.inject(:merge)))
        else
            @P = P.new(l[:P], pseed, props)
            @B = B.new(l[:B], bscale, seed_list.inject(:merge).merge(props))
        end
        @A = A.new(l[:A], aseed, seed_list)
        @I = I.new(l[:I], iscale)
    end
    def renorm
        @P.renorm
        @A.renorm
        @B.renorm
        @I.renorm
    end
    def update(p, a, n, changes)                    
        @P.update(p, n, changes[:P])
        @A.update(a, n, changes[:A])
        @B.update(n, changes[:B])
        @I.update(n, changes[:I])
    end
    def rand_init
        @P.rand_init
        @A.rand_init
        @B.rand_init
        @I.rand_init
    end
end


class SingleEta < Counter
    def initialize(l)
        super(l)
        @count = Array.new($const[:pos], 0)
        @logprob = Array.new($const[:pos], 0)
    end
    def clear
        @count = Array.new($const[:pos], 0)
    end
    def renorm
        #@logprob = @count.to_enum(:each_with_index).collect{|pos,i|
        @logprob = @count.truncate($const[:hacks][:truncate][:eta]).to_enum(:each_with_index).collect{|pos,i|
            digamma(pos + @lambda)
        }.lognorm
        clear()
    end
    def update(n, change)
        @mutex.synchronize {
            @count[n] += change
        }
    end
    def rand_init
        @logprob = @logprob.to_enum(:each_with_index).collect{|l,i| 
            Math.log(@lambda + INIT_CONST * rand)
        }.lognorm
    end
end


class Eta
    attr_accessor :P, :A, :B, :I
    def initialize(l)
        @P = SingleEta.new(l[:P])
        @B = SingleEta.new(l[:B])
        @A = SingleEta.new(l[:A])
        @I = SingleEta.new(l[:I])
    end
    def renorm
        @P.renorm
        @A.renorm
        @B.renorm
        @I.renorm
    end
    def update(n, changes)                    
        @P.update(n, changes[:P])
        @A.update(n, changes[:A])
        @B.update(n, changes[:B])
        @I.update(n, changes[:I])
    end
    def rand_init
        @P.rand_init
        @A.rand_init
        @B.rand_init
        @I.rand_init
    end
end

class SuffStats
    attr_accessor :P_count,:A_count,:B_count,:I_count,:Pe_count,:Ae_count,:Be_count,:Ie_count,:tau_count, :spill_counts
    def initialize()
        if not $const[:hacks][:sep_props]
            @P_count = Array.new($const[:clusters]){ 
                Array.new($const[:words], 0) }
        end
        @spill_counts = Array.new($const[:words],0)
        @A_count = Array.new($const[:types]){ 
            Array.new($const[:words], 0) }
        @B_count = Array.new($const[:words],0)
        @I_count = Array.new($const[:words],0)
        @tau_count = {
            :start => {:P => 0, :A => 0, :B => 0, :I => 0},
            :P => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :A => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :B => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
            :I => {:P => 0, :A => 0, :B => 0, :I => 0, :end => 0},
        }
        @Pe_count = Array.new($const[:pos],0)
        @Ae_count = Array.new($const[:pos],0)
        @Be_count = Array.new($const[:pos],0)
        @Ie_count = Array.new($const[:pos],0)
    end
    def clear()
        if not $const[:hacks][:sep_props]
            @P_count.each_with_index{|clust,i|
                clust.each_with_index{|word,j|
                    @P_count[i][j] = 0
                }
            }
        end
        @A_count.each_with_index{|att,i|
            att.each_with_index{|word,j|
                @A_count[i][j] = 0
            }
        }
        @B_count.each_with_index{|word,i|
            @B_count[i] = 0
        }
        @I_count.each_with_index{|word,i|
            @I_count[i] = 0
        }
        @spill_counts.each_with_index{|word,i|
            @spill_counts[i] = 0
        }
        @tau_count.each{|z1,counts|
            counts.each{|z2,value|
                @tau_count[z1][z2] = 0
            }
        }
        @Pe_count = @Pe_count.collect{|pos| 0}
        @Ae_count = @Ae_count.collect{|pos| 0}
        @Be_count = @Be_count.collect{|pos| 0}
        @Ie_count = @Ie_count.collect{|pos| 0}
    end
    def update(rest, snip, n, m, word)
        changes = $Zw.values[rest][snip][word]
        if $const[:hacks][:sep_props]
            $properties.values[rest][snip].each_with_index{|pp,p|
                $theta.P.update(rest,p,n,pp * changes[:P])
                #@spill_counts[n] += $const[:hacks][:spillover] * change
            }
        else
            $properties.values[rest][snip].each_with_index{|pp,p|
                @P_count[p][n] += pp * changes[:P]
                #@spill_counts[n] += $const[:hacks][:spillover] * change
            }
        end
        $attitudes.values[rest][snip].each_with_index{|pa,a|
            @A_count[a][n] += pa * changes[:A]
        }
        @B_count[n] += changes[:B]
        @I_count[n] += changes[:I]

        @Pe_count[m] += changes[:P]
        @Ae_count[m] += changes[:A]
        @Be_count[m] += changes[:B]
        @Ie_count[m] += changes[:I]

        #sticky values for HMM
        if word > 0
            $Zw.values[rest][snip][word-1].each{|z1,pz1|
                changes.each{|z2,pz2| @tau_count[z1][z2] += pz1 * pz2 } 
            }
        else
            changes.each{|z2,pz2| @tau_count[:start][z2] += pz2 } 
        end

        if word < $const[:snippet_words][rest][snip] - 1
            $Zw.values[rest][snip][word+1].each{|z2,pz2|
                changes.each{|z1,pz1| @tau_count[z1][z2] += pz1 * pz2 } 
            }
        else
            changes.each{|z1,pz1| @tau_count[z1][:end] += pz1 } 
        end 
    end
    def combinedUpdate()
        if not $const[:hacks][:sep_props]
            @P_count.each_with_index{|words,p|
                words.each_with_index{|count,n|
                    $theta.P.update(p,n,count)
                }
            }
        end
        @A_count.each_with_index{|words,a|
            words.each_with_index{|count,n|
                $theta.A.update(a,n,count)
            }
        }
        @B_count.each_with_index{|count,n|
            $theta.B.update(n,count)
        }
        @I_count.each_with_index{|count,n|
            $theta.I.update(n,count)
        }
        if $const[:hacks][:spillover] > 0
            $theta.P.update_spillover(@spill_counts)
        end
        @tau_count.each{|z1,counts|
            counts.each{|z2,value|
                $tau.update(z1,z2,value)
            }
        }

        @Pe_count.each_with_index{|count,i| $eta.P.update(i,count)}
        @Ae_count.each_with_index{|count,i| $eta.A.update(i,count)}
        @Be_count.each_with_index{|count,i| $eta.B.update(i,count)}
        @Ie_count.each_with_index{|count,i| $eta.I.update(i,count)}
    end
end


class EStepRunner
    attr_accessor :indexes, :suffStats
    def initialize(indexes)
        @indexes = indexes
        @suffStats = SuffStats.new
    end
    def run()

        suffStats.clear()

        indexes.each {|rest|

            (0..$const[:snippets][rest]-1).each{|snip|
        
                # sample new properties for the restaurant
                $properties.sample(rest,snip)

                # and for each property, sample new attitudes
                $attitudes.sample(rest,snip)
        
                (0..$const[:snippet_words][rest][snip]-1).each{|word|

                    w = $words[rest][snip][word]
                    n = $w_index[w]
                    pos = $pos[rest][snip][word]
                    m = $pos_index[pos]

                            
                    # then, sample new Zw for each word
                    $Zw.sample(rest,snip,n,m,word)
                }

                # update word suff stats when complete (because of hmm)
                (0..$const[:snippet_words][rest][snip]-1).each{|word|
                    w = $words[rest][snip][word]
                    n = $w_index[w]
                    pos = $pos[rest][snip][word]
                    m = $pos_index[pos]

                    suffStats.update(rest,snip,n,m,word)
                }

                # update suff stats
                $properties.update(rest,snip)
                $attitudes.update(rest,snip)
            }
        }

        suffStats.combinedUpdate()
    end
end

def random_init()
    $psi.rand_init
    $phi.rand_init
    $theta.rand_init
    $tau.rand_init
    $eta.rand_init
end


def data_likelihood()
    likelihood = 0.0
    $words.each_with_index{|snips,rest|
        snips.each_with_index{|words,snip|
            $properties.values[rest][snip].each_with_index{|pp,p|
                prop_like = $psi.logprob[rest][p]
                $attitudes.values[rest][snip].each_with_index{|pa,a|
                    att_like = $phi.logprob[rest][p][a]
                    words.each_with_index{|w,word|
                        n = $w_index[w]
                        $Zw.values[rest][snip][word].each{|z,pz|
                            case z
                              when :P
                                if ($const[:hacks][:sep_props])
                                  att_like += pz * $theta.P.logprob[rest][p][n]
                                else
                                  att_like += pz * $theta.P.logprob[p][n]
                                end
                              when :A then att_like += pz * $theta.A.logprob[a][n]
                              when :B then att_like += pz * $theta.B.logprob[n]
                              when :I then att_like += pz * $theta.I.logprob[n]
                            end
                        }
                    }
                    prop_like += pa * att_like
                }
                likelihood += pp * prop_like
            }
        }
    }
    return likelihood
end


def output(out,i)
    if out

        outputs = 
            (0..$const[:restaurants]-1).collect{|rest|
                { "index" => rest,
                  "file" => $files[rest],
                  "snippets" => 
                (0..$const[:snippets][rest]-1).collect{|snip|
                    { "properties" => $properties.values[rest][snip].round_to(4),
                      "attitudes" => $attitudes.values[rest][snip].round_to(4),
                      "index" => snip,
                      "text" => $snippets[rest][snip],
                      "words" => 
                    (0..$const[:snippet_words][rest][snip]-1).collect{|word|
                        w = $words[rest][snip][word]
                        n = $w_index[w]          
                        {
                            "index" => word,
                            "word" => $words[rest][snip][word],
                            "tag" => $pos[rest][snip][word],
                            "Zw" => $Zw.values[rest][snip][word].round_to(4),
                            "theta" => {
                                "P" => $const[:hacks][:sep_props] ? 
                                    $theta.P.logprob[rest].collect{|p| 
                                        Math.exp(p[n]).round_to(4)} : 
                                    $theta.P.logprob.collect{|p| 
                                        Math.exp(p[n]).round_to(4)}, 
                                "A" => $theta.A.logprob.collect{|a| 
                                    Math.exp(a[n]).round_to(4)},
                                #{
                                #    :positive => $theta.A.logprob[:positive][n],
                                #    :negative => $theta.A.logprob[:negative][n] 
                                #}.exp.round_to(4),
                                "B" => Math.exp($theta.B.logprob[n]).round_to(4),
                                "I" => Math.exp($theta.I.logprob[n]).round_to(4)
                            }
                        }
                    }
                    }
                }
                }
            }
        open("#{out}/#{i}.words","w"){|file| 
            outputs.each_with_index{|output,idx|
                file.syswrite(output.to_json + "\n")
            }
        }

        if $const[:hacks][:sep_props]
            open("#{out}/#{i}.properties", "w"){|file|
                $theta.P.logprob.each_with_index{|rest,j|
                    file.syswrite("\nRestaurant #{j}\n\n")
                    rest.each_with_index{|cluster,k|
                        words = cluster.each_with_index.sort{|a,b| 
                            b<=>a}.collect{|c| 
                                " %s:%0.4f " % [$w_name[c[1]],c[0]]
                        }[0..100]
                        file.syswrite(words.to_json + "\n")
                    }
                }
            }
        else
            FasterCSV.open("#{out}/#{i}.properties", "w"){|csv|
                $theta.P.logprob.each_with_index{|cluster,j|
                    csv << cluster.each_with_index.sort{|a,b| b<=>a}.collect{|c| 
                        " %s:%0.4f " % [$w_name[c[1]],c[0]]
                    }[0..100]
                }
            }
        end
    
        FasterCSV.open("#{out}/#{i}.attitudes", "w"){|csv|
            $theta.A.logprob.each_with_index{|cluster,j|
                csv << cluster.each_with_index.sort{|a,b| b<=>a}.collect{|c| 
                    " %s:%0.4f " % [$w_name[c[1]],c[0]]
                }[0..100]
            }
        }
    
        FasterCSV.open("#{out}/#{i}.background", "w"){|csv|
            csv << $theta.B.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c| 
                " %s:%0.4f " % [$w_name[c[1]],c[0]]
            }[0..200]
        }

        FasterCSV.open("#{out}/#{i}.ignore", "w"){|csv|
            csv << $theta.I.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c| 
                " %s:%0.4f " % [$w_name[c[1]],c[0]]
            }[0..200]
        }

        FasterCSV.open("#{out}/#{i}.eta","w"){|csv|
            csv << $eta.P.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c|
                " %s:%0.4f " % [$pos_name[c[1]],c[0]]
            }
            csv << $eta.A.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c|
                " %s:%0.4f " % [$pos_name[c[1]],c[0]]
            }
            csv << $eta.B.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c|
                " %s:%0.4f " % [$pos_name[c[1]],c[0]]
            }
            csv << $eta.I.logprob.each_with_index.sort{|a,b| b<=>a}.collect{|c|
                " %s:%0.4f " % [$pos_name[c[1]],c[0]]
            }
        }

        open("#{out}/#{i}.tau", "w"){|f|
            f.syswrite($tau.inspect + "\n")
        }
    
    end
end

$properties = Properties.new
$attitudes = Attitudes.new
$Zw = Zw.new
#$words = snippets.collect{|r| r.collect{|s| s.gsub(/(\w)(\W)/,"\\1 \\2").gsub(/(\W)(\w)/,"\\1 \\2").split.collect{|w| w.downcase}}}

$theta = Theta.new({:P => $const[:lambda][:P],:A => $const[:lambda][:A],:B => $const[:lambda][:B]},$const[:seed][:A],seed_list,$const[:seed][:P],props_seed,$const[:seed][:B],$const[:seed][:I])
$psi = Psi.new($const[:lambda][:psi])
$phi = Phi.new($const[:lambda][:phi])
$tau = Tau.new($const[:lambda][:tau], $const[:hmm], $const[:seed][:tau])
$eta = Eta.new({:P => $const[:lambda][:eta],:A => $const[:lambda][:eta],:B => $const[:lambda][:eta],:I => $const[:lambda][:eta]})

$temp = nil

rest_split = (0..$const[:restaurants]-1).to_a.split(PROCS)

splits = rest_split.collect{|indexes| EStepRunner.new(indexes)}

# initialize

time = Time.now.localtime

$stderr.print "Using #{PROCS} cores\n"
log.debug("Using #{PROCS} cores")
$stderr.print "\rinitializing\t#{time.strftime("%X")}\e[K"
log.debug("initializing\t#{time.strftime("%X")}")

random_init()

output(out,0)

# sample

(1..$const[:iters]).each{|i| 

    if i.modulo($const[:temp_step]) == 0
        $temp = ($temp.nil? ? 1 : $temp) + $const[:temp_incr]
    end

    #logprob = i > 1 ? data_likelihood : 0.0
    oldtime = time
    time = Time.now.localtime
    #$stderr.puts "iter #{i}\t#{time}\t#{logprob}"
    $stderr.print "\riter %d  %s (%0.2f)\e[K" % [i,time.strftime("%X"),oldtime-time]
    log.debug("iter %d  %s (%0.2f)" % [i,time.strftime("%X"),oldtime-time])

    threads = splits.collect{|thread| java.lang.Thread.new(thread) }
    threads.each{|thread| thread.start }
    threads.each{|thread| thread.join }

    $theta.renorm
    $psi.renorm
    $phi.renorm
    $tau.renorm
    $eta.renorm

    if i.modulo($const[:output_freq]) == 0 or i == 1
        output(out,i)
    end

}

$stderr.print "\n"


#    end
#    
#    JRubyProf.print_flat_text(result, "flat.txt")
#    JRubyProf.print_graph_text(result, "graph.txt")
#    JRubyProf.print_graph_html(result, "graph.html")
#    JRubyProf.print_call_tree(result, "call_tree.txt")
#    JRubyProf.print_tree_html(result, "call_tree.html")
#
