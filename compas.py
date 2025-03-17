import xgboost as xgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel



# subset race
def make_subsets(test_size,val_size,df,random_state, decile=True): 
    df=df[(df['race']=='African-American') | (df['race']=='Caucasian')]
    y=df['y']
    df= df.drop('y', axis=1)
   
   
    Xdegree=df['c_charge_degree']
    Xdegree=np.array([1 if Xdegree.iloc[i]=='F' else 0 for i in range(len(Xdegree))])

    Xrace=df['race']
    Xrace=np.array([1 if Xrace.iloc[i]=='Caucasian' else 0 for i in range(len(Xdegree))])

    Xsex=df['sex']
    Xsex=np.array([1 if Xsex.iloc[i]=='Male' else 0 for i in range(len(Xdegree))])

    Xage=df['age']
    Xage=np.array(Xage)

    Xprior=df['priors_count']
    Xprior=np.array(Xprior)

    Xdecile=df['decile_score']

    X1=np.column_stack((Xdegree)).reshape(-1,1)
    X2=np.column_stack((Xdegree,Xrace))
    Xfull=np.column_stack((Xdegree,Xrace,Xsex,Xage,Xprior,Xdecile))
    if decile==False:
      Xfull=np.column_stack((Xdegree,Xrace,Xsex,Xage,Xprior))

# First, split the data into train and temp (temp will be split into validation and test)
    df_train, df_temp, X_train, X_temp, X1_train, X1_temp, X2_train, X2_temp, y_train, y_temp = train_test_split(df, Xfull, X1, X2, y, test_size=(val_size + test_size), random_state=random_state)
    if val_size==0:
      return df_train, df_temp, X_train, X_temp, X1_train, X1_temp, X2_train, X2_temp, y_train, y_temp 

# Then, split temp into validation and test
    df_val, df_test, X_val, X_test, X1_val, X1_test, X2_val, X2_test, y_val, y_test = train_test_split(
      df_temp, X_temp, X1_temp, X2_temp, y_temp, test_size=test_size/(val_size + test_size), random_state=random_state)
    
    
    return df_train, df_test, df_val, X_train,X_test,X_val,X1_train,X1_test,X1_val,X2_train,X2_test,X2_val,y_train, y_test,y_val


# Custom loss function 
def custom_loss(gamma,df):
  def loss_fn(x, dtrain):
      y = dtrain.get_label()
      grad = np.array((df['race']=='African-American'))*gamma*(np.exp(x)/(np.exp(x) + 1) - np.exp(2*x)/(np.exp(x) + 1)**2) + (-y + np.exp(x)/(np.exp(x) + 1))*(2*np.exp(x)/(np.exp(x) + 1) - 2*np.exp(2*x)/(np.exp(x) + 1)**2)
      hess = np.array((df['race']=='African-American'))*gamma*(np.exp(x)/(np.exp(x) + 1) - 3*np.exp(2*x)/(np.exp(x) + 1)**2 + 2*np.exp(3*x)/(np.exp(x) + 1)**3) + (-y + np.exp(x)/(np.exp(x) + 1))*(2*np.exp(x)/(np.exp(x) + 1) - 6*np.exp(2*x)/(np.exp(x) + 1)**2 + 4*np.exp(3*x)/(np.exp(x) + 1)**3) + (np.exp(x)/(np.exp(x) + 1) - np.exp(2*x)/(np.exp(x) + 1)**2)*(2*np.exp(x)/(np.exp(x) + 1) - 2*np.exp(2*x)/(np.exp(x) + 1)**2)
      return grad, hess
  return loss_fn

def train_model(gamma,df_train,X_train,y_train):
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train the model with custom loss
    params = {"max_depth": 3, "eta": 0.1}
    bst = xgb.train(params, dtrain, num_boost_round=200, obj=custom_loss(gamma,df_train))

    return bst


def loss(gamma,df_train,df_test,X_train,X_test,y_train,y_test):
  bst=train_model(gamma=gamma,df_train=df_train,X_train=X_train,y_train=y_train)
  nhat =bst.predict(xgb.DMatrix(X_test))
  yhat=np.exp(nhat)/(1+np.exp(nhat))
  brier=sum((y_test-yhat)**2)
  bias=gamma*sum(yhat[df_test['race']=='African-American'])

  brier_baseline=sum((y_test-np.mean(y_train))**2)
  return (brier+bias,yhat,brier,brier_baseline)




def voi(gamma,df_train,df_test,X1_train,X1_test,X2_train,X2_test,y_train,y_test):

  loss1=loss(gamma,df_train,df_test,X1_train,X1_test,y_train,y_test)[0]
  loss2=loss(gamma,df_train,df_test,X2_train,X2_test,y_train,y_test)[0]

  voi=loss1-loss2
  return voi

def plot_hist(pred, race, name):
    mpl.rcParams.update(mpl.rcParamsDefault)
    # Clear any previous figures
    plt.close('all')

    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Set font size
    font_size = 16  # Change this to adjust overall font size
    plt.rc('axes', titlesize=font_size)    # Title font size
    plt.rc('axes', labelsize=font_size)    # Axis label font size
    plt.rc('xtick', labelsize=font_size)   # X-axis tick font size
    plt.rc('ytick', labelsize=font_size)   # Y-axis tick font size
    plt.rc('legend', fontsize=font_size)   # Legend font size

    data = {'pred': pred, 'Race': race}
    df = pd.DataFrame(data)
    bins = np.linspace(0, 1, 11)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='pred', hue='Race', bins=bins, kde=False, element='step', stat='density', common_norm=False)

    plt.xlim(0, 1)
    plt.ylim(0, 2.7)

    # Add labels and title
    plt.xlabel('Predicted probability of recidivism')
    plt.ylabel('Density')
    plt.title('Density histogram of predictions using ' + name + ' utility')

    # Save the plot as an SVG file
    plt.savefig('/home/voi-aistats/' + name + '.svg', format='svg', bbox_inches='tight')

    # Close the current figure after saving
    plt.close()

    # Reset to default settings
    plt.rc('text', usetex=False)
    plt.rc('font', family='sans-serif')


# Function to run the process once and return relevant values
def run_experiment(df,state,plots=False,decile=True):
    # Generate the train-test subsets
    df_train, df_test, df_val, X_train,X_test,X_val,X1_train,X1_test,X1_val,X2_train,X2_test,X2_val,y_train, y_test,y_val= make_subsets(test_size=0.2,val_size=0.2, df=df,random_state=state,decile=decile)



    # Optimize gamma using minimize_scalar
    result = minimize_scalar(lambda gamma: voi(gamma, df_train, df_test, X1_train, X1_test, X2_train, X2_test, y_train, y_test), bounds=(0, 1), method='bounded')
    gamma = result.x
    minimum_val=result.fun
    start_val=voi(0, df_train, df_test, X1_train, X1_test, X2_train, X2_test, y_train, y_test)

  
    # predictions
    pred0 = loss(0, df_train, df_val, X_train, X_val, y_train, y_val)[1]
    pred1 = loss(gamma, df_train, df_val, X_train, X_val, y_train, y_val)[1]

    #plot histogram
    if plots==True:
      plot_hist(pred0,df_val['race'],'original')
      plot_hist(pred1,df_val['race'],'modified')
      

    # Calculate means based on race
    mean_pred0_african_american = np.mean(pred0[df_val['race'] == 'African-American'])
    mean_pred0_other = np.mean(pred0[df_val['race'] != 'African-American'])
    mean_pred1_african_american = np.mean(pred1[df_val['race'] == 'African-American'])
    mean_pred1_other = np.mean(pred1[df_val['race'] != 'African-American'])

    # Return results as a dictionary
    return {
        'gamma': gamma,
        'start_val': start_val, 
        'min_val':minimum_val,
        'mean_pred0_african_american': mean_pred0_african_american,
        'mean_pred0_other': mean_pred0_other,
        'mean_pred1_african_american': mean_pred1_african_american,
        'mean_pred1_other': mean_pred1_other,
        'brier_0': loss(0, df_train, df_val, X_train, X_val, y_train, y_val)[2],
        'brier_gamma': loss(gamma, df_train, df_val, X_train, X_val, y_train, y_val)[2],
        'brier_baseline': loss(gamma, df_train, df_val, X_train, X_val, y_train, y_val)[3]
    }

def perform_test(df):
  df_train, df_test, X_train,X_test,X1_train,X1_test,X2_train,X2_test,y_train, y_test= make_subsets(test_size=0.5,val_size=0, df=df,random_state=0)

  bst1=train_model(gamma=0,df_train=df_train,X_train=X1_train,y_train=y_train)
  nhat1 =bst1.predict(xgb.DMatrix(X1_test))
  yhat1=np.exp(nhat1)/(1+np.exp(nhat1))
  brier1=(y_test-yhat1)**2

  # find brier scores using  charge degree and race
  bst2=train_model(gamma=0,df_train=df_train,X_train=X2_train,y_train=y_train)
  nhat2 =bst2.predict(xgb.DMatrix(X2_test))
  yhat2=np.exp(nhat2)/(1+np.exp(nhat2))
  brier2=(y_test-yhat2)**2

  diff=brier1-brier2
  sum(diff)
  
  
  p_value_ttest = ttest_rel(brier1, brier2)
  
  return p_value_ttest 
  


# this is the compas dataset as provided by the Julia Fairness package: https://ashryaagr.github.io/Fairness.jl/dev/datasets/
df=pd.read_csv('/home/voi/compas.csv')


results = []
# Run the experiment 20 times and store the results
for i in range(20):
    if i==0:
      result = run_experiment(df,plots=True,state=i)
      results.append(result)
    else:
      result = run_experiment(df,state=i)
      results.append(result)



results_df= pd.DataFrame(results)

#perform test for VoI
perform_test(df)


# results for tables
np.mean(results_df['gamma'])
np.min(results_df['gamma'])
np.max(results_df['gamma'])

np.mean(results_df['mean_pred0_african_american'])
np.min(results_df['mean_pred0_african_american'])
np.max(results_df['mean_pred0_african_american'])

np.mean(results_df['mean_pred1_african_american'])
np.min(results_df['mean_pred1_african_american'])
np.max(results_df['mean_pred1_african_american'])


np.mean(results_df['mean_pred0_other'])
np.min(results_df['mean_pred0_other'])
np.max(results_df['mean_pred0_other'])

np.mean(results_df['mean_pred1_other'])
np.min(results_df['mean_pred1_other'])
np.max(results_df['mean_pred1_other'])


N=make_subsets(test_size=0.2,val_size=0.2, df=df,random_state=0)[2].shape[0]
np.mean(results_df['brier_0'])/N
np.min(results_df['brier_0'])/N
np.max(results_df['brier_0'])/N


np.mean(results_df['brier_gamma'])/N
np.min(results_df['brier_gamma'])/N
np.max(results_df['brier_gamma'])/N


np.mean(results_df['brier_baseline'])/N
np.min(results_df['brier_baseline'])/N
np.max(results_df['brier_baseline'])/N

