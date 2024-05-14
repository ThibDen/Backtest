import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import numpy as np
from numba import jit
import imageio
import io
from IPython.display import Image, display


#Enter here the portfolio you are interested in if you use yahoo finance tickers
portfolio_data = [
    {"ETF": "MSCI World", "Percent_portfolio": 60, "Ticker": '^990100-USD-STRD'},
    {"ETF": "S&P 500", "Percent_portfolio": 40, "Ticker": '^GSPC'}
]


#Enter here the portfolio you are interested in if you use CSV's. For example, here I used CSV's downloaded from Curvo. 
#Comment it out if you want to use yahoo tickets. 

portfolio_data= [{"ETF": "SPRV", "Percent_portfolio": 66, "file_name": "SPRV.csv"},
                {"ETF": "SPRX", "Percent_portfolio": 34, "file_name": "SPRX.csv"},
                #{"ETF": "JPGL", "Percent_portfolio": 50, "file_name": "JPGL.csv"},             
                 ]

months=240 #how many months is the period you want to backterst
rebalance=12 # After how many months do you want to rebalance you're portfolio.
Start_invest = 10000 #What is the initial amount invested
Monthly_DCA= 1000 #Monthly amount to DCA
Transaction_costs= 0.3 #% costs to buy/sell (for example in Belgium 0.12% tax + 0.18% broker cost aprox)
date_name= 'Datum' #What is the name of the collumn where the date is in in the CSV

end_date = pd.Timestamp.today() #Last date of the sample you want to use
start_date = end_date - pd.DateOffset(years=50) #First date of the sample you want to use, here it is 50 years.

Net_of_costs= 1- Transaction_costs/100

@jit(nopython=True) #this is too make the code more efficient
def calculate_values(months, Data,start_cap, monthly_cap,Portfolio_dist,Net_of_costs,rebalance):
    nr_etf=np.shape(Data)[1] #numbers of stocks to keep track of
    time_span_data=np.shape(Data)[0]
    Value = np.zeros((time_span_data-months ,nr_etf,months)) #simulate DCA 

    for k in range( time_span_data-months): #start simulation
        Value[k,:,0] = start_cap*Portfolio_dist*Net_of_costs  #start investment for DCA analysis
        for i in range(1,months): # iterate over the number of months
            Value[k,:,i]  = Value[k,:,i-1] *(1+Data[i+k,:]) #current value of etf's is last months value of etf's * monthly returns
            Value[k,:,i]= Value[k,:,i]+(Value[k,:,i]/np.sum(Value[k,:,i])-Portfolio_dist==np.min(Value[k,:,i]/np.sum(Value[k,:,i])-Portfolio_dist) )*monthly_cap*Net_of_costs #DCA into the fund with highest downward deviation from model portfolio
            if i % rebalance == 0: #if we are in a month where we want to rebalance
                Value[k,:,i]=Value[k,:,i]-((Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))>0)*(Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))/Net_of_costs #sell excess etf's untill exact desired %
                Value[k,:,i]=Value[k,:,i]-((Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))<0)*(Value[k,:,i]-Portfolio_dist*np.sum(Value[k,:,i]))*Net_of_costs #use that money to buy etf's you are short of, ofcourse loosing some money to transaction costs

    return np.sum(Value,axis=1)#add the values of the etf's toghetter to get portfolio value per month per simulation

def download_and_merge(portfolio_data, date_name,start_date,end_date):
    merged_df = pd.DataFrame()

    for etf in portfolio_data:
        # Download the CSV file
        df = pd.read_csv(etf['file_name'], index_col=date_name, parse_dates=True)

        # Rename the column to the ETF name
        df.columns = [etf['ETF']]

        # Merge the dataframes
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')
    mask = (merged_df.index >= start_date) & (merged_df.index <= end_date)
    return merged_df.loc[mask]



#histogram function
def create_histogram_gif(Returns, Start_invest, Monthly_DCA, portfolio_name):
    # Create a list to store each frame of the GIF
    images = []

    # Define the bins for the histogram
    bins = np.linspace((np.min(Returns)), (np.max(Returns)), 50)



    # Define the range for the y-axis (you can adjust these values as needed)
    y_min = 0

    # Loop over each year (assuming Returns is monthly data)
    for i in range(11, Returns.shape[1], 12):
        y_max=max(np.shape(Returns)[0]*(1-(2*(i-11)/np.shape(Returns)[1])),np.shape(Returns)[0]/4)
        title= f'Initial investment of {Start_invest}, DCA of {Monthly_DCA} after {i//12+1} years'


        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram with fixed bins for Returns_A
        ax.hist(Returns[:, i], bins=bins, color='blue', alpha=0.5, label=f'{portfolio_name}')

        # Create histogram with fixed bins for Returns_B

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Returns (%)')
        ax.set_ylabel('Frequency')
        ax.set_ylim(y_min, y_max)
        ax.legend()

        # Save plot to a PNG file in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Add PNG file to images list
        images.append(imageio.imread(buf))

        # Close the figure
        plt.close()

    # Create GIF from images list
    gif = imageio.mimwrite(imageio.RETURN_BYTES, images, 'GIF', duration=0.5)

    # Display GIF in Jupyter Notebook
    display(Image(data=gif))

#function to generate tables
def Table(data,months, title_port, table_text):
    percentiles = [1, 10, 25, 50, 75, 90, 99]
    labels = ['Worst 1%', 'Worst 10%', 'Worst 25%', 'Median', 'Best 25%', 'Best 10%', 'Best 1%']
    years = [1, 3, 5, 10, 15, 20,30,40]

    # Convert years to months
    indice= [year * 12-1 for year in years]
    indice = [i for i in indice if i < months] #select only years within the month range

    # Initialize an empty dataframe to store the results
    df = pd.DataFrame(index=labels)
    # Calculate the percentiles for each year
    for month in indice:
        percentile_values = np.percentile(data[:, month], percentiles, axis=0)
        df[f'After {month//12+1} years'] = percentile_values

    # Define the title
    print(f'{title_port}: {table_text}')

    # Apply the float format to each cell in the DataFrame
    if df.iloc[1,1]>3:
    # Print the DataFrame in a nice table format
        print(df.to_markdown(floatfmt='.2f'))
    else:
        print(df.to_markdown(floatfmt='.4f'))
  


# The main function
def calculate_returns_and_plot(months, Start_invest, Monthly_DCA,Net_of_costs,rebalance, portfolio_data,start_date,end_date,date_name):
    #retrieve relevant data per portfolio

    if  np.sum(np.array([["file_name" in etf] for etf in portfolio_data]))==0 :
        data = yf.download(list([d["Ticker"] for d in portfolio_data]), interval='1mo',start=start_date, end=end_date)['Close']
    else :
        data= download_and_merge(portfolio_data,date_name, start_date, end_date)
    data=data.dropna()

    data=data.pct_change()

    data=data.dropna()
    data= np.array(data, dtype=np.float64)        
    Portfolio_dist=np.array([etf['Percent_portfolio'] for etf in portfolio_data])/100 

    Value=calculate_values(months, data, Start_invest, Monthly_DCA, Portfolio_dist,Net_of_costs,rebalance)

    #make the portfolio names for the grpahs 
    title_port =  f'{"/".join([d["ETF"] for d in portfolio_data])}:{"/".join([str(d["Percent_portfolio"]) for d in portfolio_data])}'

    #calcualte the invested amount
    invested = Start_invest + Monthly_DCA * np.arange(months)

    #calcualted the percentiles for the graphs
    percentiles = [1, 10, 25,50, 75, 90, 99]
    labels = ['Worst 1%', 'Worst 10%', 'Worst 25%', 'Median','Best 25%', 'Best 10%', 'Best 1%']
    colors = ['black', 'red', 'orange', 'Blue','lime', 'green', 'darkgreen']

    returns= [(np.nanpercentile(Value, p, axis=0) - invested) / invested for p in percentiles]

    chance_of_profit = np.nansum(Value > invested, axis=0) / (np.shape(Value)[0] - np.nansum(np.isnan(Value), axis=0))

    # Prepend initial_point to each time series
    returns = [np.insert(r, 0, 0) for r in returns]

    year_t = np.arange(len(returns[0])) / 12  # Assuming 'months' represents the time in months

# Calculate the maximum of the 99th percentile of Returns_A and Returns_B

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot for returns_A
    for r, label, color in zip(returns, labels, colors):
        ax.plot(year_t, r * 100, label=label, color=color)
    ax.set_xlabel('Years')
    ax.set_ylabel('Returns on money invested(%)')
    ax.set_title(f"Returns with {title_port}")
    ax.legend()
    ax.grid(True)

    # Plot for returns_B

    plt.show()


    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for chance_of_profit_A
    ax.plot(year_t[1:], chance_of_profit * 100, label=title_port, color='blue')

    # Plot for chance_of_profit_B

    ax.set_xlabel('Years')
    ax.set_ylabel('Chance of profit (%)')
    ax.set_title(f'Probability of profit with DCA of {Monthly_DCA}]', fontsize=16)
    ax.legend()
    ax.grid(True)

    plt.show()
    create_histogram_gif((Value/invested-1)*100, Start_invest, Monthly_DCA, title_port)
    table_text=f'End value with DCA of {Monthly_DCA}, net of inlation'
    Table(Value,months, title_port, table_text)

# Call the function
calculate_returns_and_plot(months, Start_invest, Monthly_DCA,Net_of_costs,rebalance, portfolio_data,start_date,end_date,date_name)

