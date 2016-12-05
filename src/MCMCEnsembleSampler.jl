module MCMCEnsembleSampler

import StatsBase
export mcmc

function mcmc(f::Function, max_iter::Int, n_walkers::Int, n_dim::Int, init_range::Array, jump::String)

  max_iter > n_walkers || error("max_iter must be larger than n_walkers")
  (jump != "s_m" || jump != "d_e") || error("Invalid selection for jump: select either s_m or d_e.")

  # initial values
  chain_length = div(max_iter, n_walkers)
  sum_log_p = zeros(chain_length)

  log_p = Array{Float64}(n_walkers, chain_length)
  log_p_old = Array{Float64}(n_walkers)

  ensemble_old = rand(n_walkers, n_dim) * (init_range[2] - init_range[1]) + init_range[1]
 #transponieren ensemble_old, ensemble_new
  ensemble_new = Array{Float64}(n_dim)
  x_chain = Array{Float64}(n_walkers, chain_length, n_dim)

  for k in 1:n_walkers
      log_p_old[k] = f(ensemble_old[k,:])
  end

  log_p[:,1] = log_p_old
  x_chain[:, 1, :] = ensemble_old

  # the loop
  for l in 2:chain_length
    for n in 1:n_walkers

      if (jump == "s_m")
        ensemble_new = s_m(n_walkers, ensemble_old, n_dim, n)
      elseif (jump == "d_e")
        ensemble_new = d_e(n_walkers, ensemble_old, n_dim, n, l)
      end

      log_p_new = f(ensemble_new)

      acc = exp(log_p_new - log_p_old[n])
      if (acc > rand())
        x_chain[n,l,:] = ensemble_new
        ensemble_old[n,:] = ensemble_new
        log_p[n,l] = log_p_new
        log_p_old[n] = log_p_new
      else
        x_chain[n,l,:] = ensemble_old[n,:]
        log_p[n,l,:] = log_p_old[n]
      end

    end
  end
  return x_chain, log_p
end


function s_m(n_walkers::Int, ensemble_old::Array{Float64}, n_dim::Int, n::Int)

  z = ((rand()+1)^2)/2
  a = StatsBase.sample([i for i in 1:n_walkers if i != n])
  par_active = ensemble_old[a,:]

  return par_active + z^(n_dim-1) * (ensemble_old[n,:] - par_active)
end


function d_e(n_walkers::Int, ensemble_old::Array{Float64}, n_dim::Int, n::Int, l::Int)

  z = 2.38 / sqrt(2 * n_dim)
  if (l % 10 == 0)
    z=1
  end

  a = StatsBase.sample([i for i in 1:n_walkers if i != n])
  b = StatsBase.sample([i for i in 1:n_walkers if (i != n && i != a)])

  par_active_1 = ensemble_old[a,:]
  par_active_2 = ensemble_old[b,:]

  return ensemble_old[n,:] + z*(par_active_1 - par_active_2)

end


end # module
