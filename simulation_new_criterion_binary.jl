using Random
using Optim
using Distributions
using GLM
using DataFrames



U(w1,w2,w3;data)=w1*data.S+w2*data.N+w3*data.M

function policy_l(w1,w2,w3;models,data)

	#fit model with (Mp)
	p0=models["M_Mp"]*U(w1,w2,w3,data=data)	

	#fit model with (S,Mp)
	p1=models["M_SMp"]*U(w1,w2,w3,data=data)
	
	#calculate policy difference

	l=sqrt((p1[1]-p0[1])^2+(p1[2])^2+(p1[3]-p0[2])^2)
	return l
end


function utility_l(w1,w2,w3;thetas,data)
	
	old=thetas["thetaNU"]*data.N+thetas["thetaMU"]*data.M
	new=U(w1,w2,w3,data=data)

	l=mean((old.-new).^2)
	
	return l
end


function loss(w1,w2,w3,thetas,models,data,weight)
	
	return utility_l(w1,w2,w3,thetas=thetas,data=data)+weight*policy_l(w1,w2,w3,models=models,data=data)
end



function numeric_solution(thetas,models,data)

sol=[1,1,1]
solution_found=false
weight=1

while solution_found==false
	res=optimize(w->loss(w[1],w[2],w[3],thetas,models,data,weight),[1.,1.,1.],LBFGS(),autodiff=:forward, Optim.Options(iterations=1000))

	sol=Optim.minimizer(res)
	if policy_l(sol[1],sol[2],sol[3],models=models,data=data)>0.0001
		weight=weight*2
	else
		solution_found=true	
	end 
end

return sol
end

function analytic_solution(thetaSNp,thetaSMp,thetaNpN,thetaMpM,thetaNU,thetaMU)
return (-thetaNU*thetaSNp*thetaNpN,thetaNU, thetaMU+thetaSNp*thetaNpN*thetaSMp*thetaMpM*thetaNU/(1+thetaMpM^2+(thetaSMp*thetaMpM)^2))
end

function estimate_appropriate(thetaSNp,thetaSMp,thetaNpN,thetaMpM,thetaNU,thetaMU,n,m)

thetas=Dict("thetaSNp"=>thetaSNp,"thetaSMp"=>thetaSMp,"thetaNpN"=>thetaNpN,"thetaMpM"=>thetaMpM,
			"thetaNU"=>thetaNU,"thetaMU"=>thetaMU)

estimates=Array{Vector{Float64}}(undef,m,1)
for i=1:m
	d=Normal()


	E_S=rand(d,n)
	E_Np=rand(d,n)
	E_Mp=rand(d,n)
	E_N=rand(d,n)
	E_M=rand(d,n)

	S=[if x>0 1 else -1 end for x in E_S]
	Np=thetaSNp*S.+E_Np
	Mp=thetaSMp*S.+E_Mp
	M=thetaMpM*Mp.+E_M
	N=thetaNpN*Np.+E_N



	X=hcat(repeat([1],outer=n),Mp)
	M_Mp=inv(X'*X)*X'

	X=hcat(repeat([1],outer=n),S,Mp)
	M_SMp=inv(X'*X)*X'

	models=Dict("M_SMp"=>M_SMp,"M_Mp"=>M_Mp)
	data=DataFrame(S=S,N=N,M=M)

	estimates[i]=numeric_solution(thetas,models,data)
end


return estimates
end


Random.seed!(1)

est1=estimate_appropriate(1,2,3,4,5,6,10000,100)

est1=[[x[j] for x in est1] for j=1:3]

an1=analytic_solution(1,2,3,4,5,6)


d1=DataFrame(w1=est1[1][:],w2=est1[2][:],w3=est1[3][:])

mean(d1.w1)
quantile(d1.w1,[0.05,0.95])

mean(d1.w2)
quantile(d1.w2,[0.05,0.95])

mean(d1.w3)
quantile(d1.w3,[0.05,0.95])




est2=estimate_appropriate(6,5,4,3,2,1,10000,100)

est2=[[x[j] for x in est2] for j=1:3]

an2=analytic_solution(6,5,4,3,2,1)


d2=DataFrame(w1=est2[1][:],w2=est2[2][:],w3=est2[3][:])

mean(d2.w1)
quantile(d2.w1,[0.05,0.95])

mean(d2.w2)
quantile(d2.w2,[0.05,0.95])

mean(d2.w3)
quantile(d2.w3,[0.05,0.95])



