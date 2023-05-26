using Optim
using Random
using Distributions
using GLM
using DataFrames


function generate_data(n)
	
	E_S=rand(Uniform(),n)
	S=[if x>1/3 1 else -1 end for x in E_S]

	#effort
	E=rand(Normal(0,1),n);

	#test
	T=E.+rand(Normal(0.3,1),n).*S;
	
	#resume
	R=E.+rand(Normal(1,1),n).*S;

	#graduation
	Y=E.+0.5.*R+0.5*T.+rand(Normal(0,2),n)
	Y=1*(Y .> quantile(Y,0.7))
	mean(Y[S.==1])
	mean(Y[S.==-1])


	##estimate expected graduation rate
	data=DataFrame(Y=Y,R=R,T=T,S=1(S.>0))
	return(data)
end


function VoI(test,w1,pred_Y_T,pred_S_T,pred_Y_TS)
	#first fit with only T
	
	threshold=sort(pred_Y_T .+ w1*pred_S_T,rev=true)[Int64(length(pred_Y_T)/2)]
	utility_T=sum((test.Y .+ w1*test.S)[pred_Y_T .+ w1*pred_S_T .> threshold])

	#first fit with  T and S

	threshold=sort(pred_Y_TS .+ w1*test.S,rev=true)[Int64(length(pred_Y_TS)/2)]
	utility_TS=sum((test.Y .+w1*test.S)[pred_Y_TS .+ w1*test.S .> threshold])

	return utility_TS-utility_T
end

function calculate_results()

#training
train=generate_data(100000)


#test
test=generate_data(100000)


#validation
val=generate_data(100000)


probit_Y_T=glm(@formula(Y~T),train,Binomial(),ProbitLink())
pred_Y_T=predict(probit_Y_T,test)

probit_S_T=glm(@formula(S~T),train,Binomial(),ProbitLink())
pred_S_T=predict(probit_S_T,test)

probit_Y_TS=glm(@formula(Y~T*S),train,Binomial(),ProbitLink())
pred_Y_TS=predict(probit_Y_TS,test)


res=optimize(x->VoI(test,x[1],pred_Y_T,pred_S_T,pred_Y_TS),[0.],LBFGS(),autodiff=:forward)
w1=Optim.minimizer(res)
minimum=Optim.minimum(res)

probit=glm(@formula(Y~R*T*S),train,Binomial(),ProbitLink())
pred=predict(probit,val);

###college attainment under original utility

threshold=sort(pred,rev=true)[Int64(length(pred)/2)]
v1=sum((val.Y)[pred .>= threshold])

## minority admitted under original utility
v2=sum((pred .>= threshold) .* (val.S.==0))

## minority attainment under original utility
v3=sum((val.Y)[(pred .>= threshold) .* (val.S.==0)])

##college attainment under modified utility

threshold=sort(pred .+ w1.*val.S,rev=true)[Int64(length(pred)/2)]
v4=sum((val.Y)[pred .+ w1.*val.S .>= threshold])

## minority admitted under modified utility
v5=sum((pred .+ w1.*val.S .>= threshold) .* (val.S.==0))

## minority attainment under original utility
v6=sum((val.Y)[(pred .+ w1.*val.S .>= threshold) .* (val.S.==0)])

return (minimum,w1,v1,v2,v3,v4,v5,v6)
end


Random.seed!(1);
results=[]
for i=1:100
	r=calculate_results()
	push!(results,r)
end

f=results

#average number less graduating
mean([f[i][3]-f[i][6] for i=1:length(f)])
quantile(([f[i][3]-f[i][6] for i=1:length(f)]),0.05)
quantile(([f[i][3]-f[i][6] for i=1:length(f)]),0.95)


#average number more addmitted from minority group
mean([f[i][7]-f[i][4] for i=1:length(f)])
quantile(([f[i][7]-f[i][4] for i=1:length(f)]),0.05)
quantile(([f[i][7]-f[i][4] for i=1:length(f)]),0.95)

#average number more graduating from minority group
mean([f[i][8]-f[i][5] for i=1:length(f)])
quantile(([f[i][8]-f[i][5] for i=1:length(f)]),0.05)
quantile(([f[i][8]-f[i][5] for i=1:length(f)]),0.95)



