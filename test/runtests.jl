using MCMCEnsembleSampler
using Base.Test

function p_log(x::Array{Float64})

  B = 0.03
  return -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2

end

function log_post_20_d_rosenbrock(xx)
  y=0.0
  for dim in 1:19
    w = -(1.0-xx[dim])^2 - 100.0*(xx[dim+1]-xx[dim]^2)^2
    y += w
  end

  return y
end

#output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])
#output = d_e_mcmc(p_log, 100, 20, 2, [-1.0,2.0])

output = mcmc(p_log, 100, 20, 2, [-1.0,2.0], "s_m")
output = mcmc(p_log, 100, 20, 2, [-1.0,2.0], "d_e")

@time for i in 1:5
  output = mcmc(log_post_20_d_rosenbrock, 100000, 20, 20, [-3.0,10.0], "s_m")
  output = mcmc(log_post_20_d_rosenbrock, 100000, 20, 20, [-3.0,10.0], "d_e")
end

# write your own tests here
# @test 1 == 2
