module MCMCEnsembleSampler

import StatsBase
export s_m_mcmc, d_e_mcmc

function s_m_mcmc(f::Function, max_iter::Int, n_walkers::Int, n_dim::Int, init_range::Array)

  max_iter > n_walkers || error("max_iter must be larger than n_walkers")

  # initial values
  chain_length = div(max_iter, n_walkers)
  sum_log_p = zeros(chain_length)

  log_p = Array{Float64}(n_walkers, chain_length)
  log_p_old = Array{Float64}(n_walkers)

  ensemble_old = rand(n_walkers, n_dim) * (init_range[2] - init_range[1]) + init_range[1]

  ensemble_new = Array{Float64}(n_walkers, n_dim)
  x_chain = Array{Float64}(n_walkers, chain_length, n_dim)

  for k in 1:n_walkers
      log_p_old[k] = f(ensemble_old[k,:])
  end

  log_p[:,1] = log_p_old
  sum_log_p[1] = sum(log_p_old[1:n_walkers])/n_walkers

  x_chain[:, 1, :] = ensemble_old

  # the loop
  for l in 2:chain_length
    for n in 1:n_walkers

      z = ((rand()+1)^2)/2
      a = StatsBase.sample([i for i in 1:n_walkers if i != n])
      par_active = ensemble_old[a,:]

      ensemble_new[n,:] = par_active + z^(n_dim-1) * (ensemble_old[n,:] - par_active)

      log_p_new = f(ensemble_new[n,:])

      acc = exp(log_p_new - log_p_old[n])
      if (acc > rand())
        x_chain[n,l,:] = ensemble_new[n,:]
        ensemble_old[n,:] = ensemble_new[n,:]
        log_p[n,l] = log_p_new
        log_p_old[n] = log_p_new
      else
        x_chain[n,l,:] = ensemble_old[n,:]
        log_p[n,l,:] = log_p_old[n]
      end

      sum_log_p[l] += log_p[n,l]/n_walkers

    end
  end
  return x_chain, log_p
end


function d_e_mcmc(f::Function, max_iter::Int, n_walkers::Int, n_dim::Int, init_range::Array)

  # initial values

  chain_length = div(max_iter, n_walkers)
  sum_log_p_d_e = zeros(Float64, chain_length)

  log_p = Array{Float64}(n_walkers, chain_length)
  log_p_old = Array{Float64}(n_walkers)

  ensemble_old = rand(n_walkers, n_dim) * (init_range[2] - init_range[1]) + init_range[1]

  ensemble_new = Array{Float64}(n_walkers, n_dim)
  x_chain = Array{Float64}(n_walkers, chain_length, n_dim)

  for k in 1:n_walkers

      log_p_old[k] = f(ensemble_old[k,:])

  end

  log_p[:,1] = log_p_old
  sum_log_p_d_e[1] = sum(log_p_old[1:n_walkers])/n_walkers

  x_chain[:, 1, :] = ensemble_old

  # the loop

  for l in 2:chain_length

    for n in 1:n_walkers

      z = 2.38 / sqrt(2 * n_dim)
      if (l % 10 == 0)
        z=1
      end

      a = StatsBase.sample(deleteat!(collect(1:1:n_walkers), n))
      b = StatsBase.sample(deleteat!(collect(1:1:n_walkers), sort([n,a])))

      par_active_1 = ensemble_old[a,:]
      par_active_2 = ensemble_old[b,:]

      ensemble_new[n,:] = ensemble_old[n,:] + z*(par_active_1 - par_active_2)

      log_p_new = f(ensemble_new[n,:])
      acc = exp(log_p_new - log_p_old[n])
      test = rand()

      if (acc > test)

        x_chain[n,l,:] = ensemble_new[n,:]
        ensemble_old[n,:] = ensemble_new[n,:]
        log_p[n,l] = log_p_new
        log_p_old[n] = log_p_new

      else

        x_chain[n,l,:] = ensemble_old[n,:]
        log_p[n,l,:] = log_p_old[n]

      end

      sum_log_p_d_e[l] = sum_log_p_d_e[l] + log_p[n,l]/n_walkers

    end

  end

  return x_chain

end


end # module
