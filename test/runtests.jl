using MCMCEnsembleSampler
using Base.Test

function p_log(x::Array{Float64})

  B = 0.03
  return -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2

end

#output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])
#output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])

output = s_m_mcmc(p_log, 100, 20, 2, [-1.0,2.0])
output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])

@time output = s_m_mcmc(p_log, 100, 20, 2, [-1.0,2.0])
@time output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])


# write your own tests here
# @test 1 == 2
