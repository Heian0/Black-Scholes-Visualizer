import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="Black-Scholes Model")

def black_scholes(s: float, k: float, r: float, t: float, sig: float, type=1):
    '''
    Calculate the theoretical price of a call/put option given the following parameters:

        - s: float - the current price of the security for this option

        - k: float - the strike price of this option

        - r: float - the risk free interest rate (assumed constant by the Black-Scholes model, although not a realistic assumption)

        - t: float - time remaining until this option contract expires

        - type: int - this type of this option, call = 1 or put = 0

        Returns price: float, the theoretical price of this option at this point in time

    '''
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    # Catch input errors
    try:

        # Call option
        if type:
            return s * norm.cdf(d1, 0, 1) - k * np.exp(-r * t) * norm.cdf(d2, 0, 1)

        # Put option
        else:
            return k * np.exp(-r * t) * norm.cdf(-d2, 0, 1) - s * norm.cdf(-d1, 0, 1)

    except:
        st.sidebar.error("Please input all option parameters!")

def delta(s: float, k: float, r: float, t: float, sig: float, type=1):
    '''
    Calculate the delta greek for this option - the rate of change between the option's price and a 1 dollar change in the underlying asset's price.
    Input parameters are the same as the Black-Scholes formula.
    '''
    
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    try:
        # Call option
        if type:
            return norm.cdf(d1, 0, 1)
        # Put option
        else:
            return -norm.cdf(-d1, 0, 1)

    except:
        st.sidebar.error("Please input all option parameters!")


def gamma(s: float, k: float, r: float, t: float, sig: float):
    '''
    Calculate the gamma greek for this option - the rate of change between the option's delta and a 1 dollar change in the underlying asset's price.
    Gamma is the derivative of delta.
    '''
    
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    # Gamma calculation is the same for both call and put options
    try:
        return norm.pdf(d1, 0, 1)/ (S * sig * np.sqrt(t))

    except:
        st.sidebar.error("Please input all option parameters!")


def theta(s: float, k: float, r: float, t: float, sig: float, type=1):
    '''
    Calculate the theta greek for this option - the rate of change between the option's price and the remaining time until the option expires.
    '''
    
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    try:
        # Call option
        if type:
            theta = - ((s * norm.pdf(d1, 0, 1) * sig) / (2 * np.sqrt(t))) - r * k * np.exp(-r*t) * norm.cdf(d2, 0, 1)

        # Put option
        else:
            theta = - ((s * norm.pdf(d1, 0, 1) * sig) / (2 * np.sqrt(t))) + r * k * np.exp(-r*t) * norm.cdf(-d2, 0, 1)

        return theta/365

    except:
        st.sidebar.error("Please input all option parameters!")


def vega(s: float, k: float, r: float, t: float, sig: float):
    '''
    Calculate the vega greek for this option - the rate of change between the option's price and the security's implied voaltility.
    '''
    
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    # Vega calculation is the same for both call and put options
    try:
        return s * np.sqrt(t) * norm.pdf(d1, 0, 1) * 0.01

    except:
        st.sidebar.error("Please input all option parameters!")


def rho(s: float, k: float, r: float, t: float, sig: float, type=1):
    '''
    Calculate the rho greek for this option - the rate of change between the option's price and a 1% change in the interest rate.
    '''
    
    # Calculate probability factors
    d1 = (np.log(s / k) + (r + sig ** 2 / 2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)

    try:
        # Call option
        if type:
            return 0.01 * k * t * np.exp(-r*t) * norm.cdf(d2, 0, 1)

        # Put option
        else:
            return 0.01 * -k * t * np.exp(-r*t) * norm.cdf(-d2, 0, 1)

    except:
        st.sidebar.error("Please input all option parameters!")


sidebar_title = st.sidebar.header("Black-Scholes Input Parameters")
space = st.sidebar.header("")
r = st.sidebar.number_input("Risk-Free Rate", min_value=0.000, max_value=1.000, step=0.001, value=0.050)
S = st.sidebar.number_input("Underlying Asset Price", min_value=1.00, step=0.10, value=100.00)
K = st.sidebar.number_input("Strike Price", min_value=1.00, step=0.10, value=100.00)
days_to_expiry = st.sidebar.number_input("Time to Expiry Date (in days)", min_value=1, step=1, value=365)
sigma = st.sidebar.number_input("Volatility", min_value=0.000, max_value=1.000, step=0.01, value=0.20)
min_spot_price = st.sidebar.number_input("Minimum Spot Price", min_value=0.000, step=1.0, value=75.0)
max_spot_price = st.sidebar.number_input("Maximum Spot Price", min_value=1.000, step=1.0, value=125.0)
min_volatility = st.sidebar.number_input("Minimum Volatility", min_value=0.00, step=0.01, value=0.10)
max_volatility = st.sidebar.number_input("Maximum Volatility", min_value=0.10, step=0.01, value=0.30)
type_input = st.sidebar.selectbox("Option Type",["Call", "Put"])

type=-1
if type_input=="Call":
    type = 1
elif type_input=="Put":
    type = 0

T = days_to_expiry/365

# Calculate probability factors
d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)


spot_prices = [i for i in range(0, int(S) + 1)]

def generate_equally_spaced_numbers(start, end):
    step = (end - start) / 9 
    return [round((start + step * i), 2) for i in range(10)]

heatmap_spot_prices = generate_equally_spaced_numbers(min_spot_price, max_spot_price)
heatmap_volatilities = generate_equally_spaced_numbers(min_volatility, max_volatility)

call_heatmap_matrix = []
put_heatmap_matrix = []

for i in range(0, 10):
    call_line = []
    put_line = []
    for j in range(0, 10):
        call_line.append(black_scholes(heatmap_spot_prices[i], K, r, T, heatmap_volatilities[j], 1))
        put_line.append(black_scholes(heatmap_spot_prices[i], K, r, T, heatmap_volatilities[j], 0))
    call_heatmap_matrix.append(call_line)
    put_heatmap_matrix.append(put_line)

prices = [black_scholes(i, K, r, T, sigma, type) for i in spot_prices]
deltas = [delta(i, K, r, T, sigma, type) for i in spot_prices]
gammas = [gamma(i, K, r, T, sigma) for i in spot_prices]
thetas = [theta(i, K, r, T, sigma, type) for i in spot_prices]
vegas = [vega(i, K, r, T, sigma) for i in spot_prices]
rhos = [rho(i, K, r, T, sigma, type) for i in spot_prices]

sns.set_style("whitegrid")

fig1, ax1 = plt.subplots()
sns.lineplot(x=spot_prices, y=prices)
ax1.set_ylabel('Option Price')
ax1.set_xlabel("Underlying Asset Price")
ax1.set_title("Option Price")

fig2, ax2 = plt.subplots()
sns.lineplot(x=spot_prices, y=deltas)
ax2.set_ylabel('Delta')
ax2.set_xlabel("Underlying Asset Price")
ax2.set_title("Delta")

fig3, ax3 = plt.subplots()
sns.lineplot(x=spot_prices, y=gammas)
ax3.set_ylabel('Gamma')
ax3.set_xlabel("Underlying Asset Price")
ax3.set_title("Gamma")

fig4, ax4 = plt.subplots()
sns.lineplot(x=spot_prices, y=thetas)
ax4.set_ylabel('Theta')
ax4.set_xlabel("Underlying Asset Price")
ax4.set_title("Theta")

fig5, ax5 = plt.subplots()
sns.lineplot(x=spot_prices, y=vegas)
ax5.set_ylabel('Vega')
ax5.set_xlabel("Underlying Asset Price")
ax5.set_title("Vega")

fig6, ax6 = plt.subplots()
sns.lineplot(x=spot_prices, y=rhos)
ax6.set_ylabel('Rho')
ax6.set_xlabel("Underlying Asset Price")
ax6.set_title("Rho")

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()


st.markdown("<h2 align='center'>Black-Scholes Options Pricer + Volatility Heatmap & Greeks Visualizer</h2>", unsafe_allow_html=True)
st.markdown("<h5 align='center'>Tim Li, Computer Science @ University of Toronto</h5>", unsafe_allow_html=True)
st.markdown("<h6 align='center'>timrui.li@mail.utoronto.ca | LinkedIn | Github", unsafe_allow_html=True)
st.header("")

inputs = np.array([S, K, T, sigma, r])
df = pd.DataFrame([inputs], columns=(["Current Security Price", 
                                                  "Strike Price",
                                                  "Time to Maturity (Years)",
                                                  "Volatility (σ)",
                                                  "Risk-Free Interest Rate"]))

st.table(df)

col1, col2, col3, col4, col5 = st.columns(5)
col2.metric("Call Price", "$" + str(round(black_scholes(S, K, r, T, sigma,type=1), 3)))
col4.metric("Put Price", "$" + str(round(black_scholes(S, K, r, T, sigma,type=0), 3)))

bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
bcol1.metric("Delta", str(round(delta(S, K, r, T, sigma,type=1), 3)))
bcol2.metric("Gamma", str(round(gamma(S, K, r, T, sigma), 3)))
bcol3.metric("Theta", str(round(theta(S, K, r, T, sigma,type=1), 3)))
bcol4.metric("Vega", str(round(vega(S, K, r, T, sigma), 3)))
bcol5.metric("Rho", str(round(rho(S, K, r, T, sigma,type=1), 3)))


# Create a heatmap plot
def create_heatmap(data, type):
    fig, ax = plt.subplots()
    sns.heatmap(data, annot=True, cmap=LinearSegmentedColormap.from_list('RedGreen', ['red', 'yellow', 'green']), fmt=".2f", xticklabels=heatmap_volatilities, yticklabels=heatmap_spot_prices, annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel('Volatility (σ)', fontsize=10)
    ax.set_ylabel('Spot Price ($)', fontsize=10)
    if type: ax.set_title("Call Heatmap", fontsize=11)
    else: ax.set_title("Put Heatmap", fontsize=11)
    return fig

st.title("Call Heatmap")

# Display heatmap
call_fig = create_heatmap(np.array(call_heatmap_matrix), 1)
st.pyplot(call_fig)


st.title("Put Heatmap")

# Display heatmap
put_fig = create_heatmap(np.array(put_heatmap_matrix), 0)
st.pyplot(put_fig)


st.header("")
st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
st.header("")
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)
st.pyplot(fig5)
st.pyplot(fig6)

