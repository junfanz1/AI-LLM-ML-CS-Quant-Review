# Risk Methodology

Compliance: SR11-7 (Supervisory Guidance on model risk management)

Prime Services front office (prime brokerage), manage market and counterparty risk. Equity prime brokerage, futures clearing, F/X. Key metric: 4-day 99% VaR number for client.

The model can cohesively model dozens of different security types and a wide variety of market scenario outcomes while producing competitive margin values. A model that could provide cross-margining capabilities at competitive levels while at the same time covering the risk. 

# Monte Carlo VaR vs. Historical VaR and Parametric VaR

MC simulates a variety of outcomes across many risk factors and finds the cases that leads to tail losses. Alternatives aren’t good, because stress/scenario risk measure are conservative for regular margining though has its place on part of an overall risk mgmt, and suitable in finding instances of potential large losses deep in the tail distribution. 

Prime Services wishes to account for a variety of potential ouutcomes that may not have been realized historically. We risk hedge funds that use hedging and arbitrage strats, it was imperative that we find cases where potential risk might lurk in strats that on the surface looked hedged. Historical VaR can only look at a certain set of historical returns and precise returns over longer periods may not be available for all different asset classes. 

Historical VaR has advantage of speed of calculation, but is not suitable for examining risks of arbitrage and hedging clients.

Parametric VaR suffers from being overly conservative or not covering risk in all potential cases. Problems are: defining large number of parameters across all different securities and still produce a competitive margin number. 

MC VaR with enough scenarios would meet the needs of covering the risk precisely and naturally.

# Product: Models

## 1. PV

Commodity/Currency Future, Commodity Futures Spread, Conversible Preferred Stock, Quanto Future, Preferred Stock, Loan and Borrow, Repo, FX Forward, Future, Deliverable Swap Futures, Swap

## 2. Black, Whaley

Commodity/Currency Future Option, Listed ADR Option, (Forward Value) Future Option, Basket Option, Right, Listed Option, OTC Option, Quanto Listed, Warrant, Fed Funds Futures Option, Currency OTC, Swaption, DECS, FX Touch/Rebate Option, FX Digital/Barrier Option, Bond Future Option


## 3. Bachelier

Commodity Futures Spread Option

## 4. HazardRate

Exchangeable, Convertible, Credit Default Swap

### PV, HazardRate

Corporate Bond (Callable Hazard Model), Floating Rate Note

## 5. Standard

ADR, Stock, DI Future Brazilian, Australian Bond Future, MM Future, Fed Funds Future, OIS Future, FRA

## 6. Inflation

Inflation Bond

## 7. HW

Bond Future, MM Future Option

### HJM (Lognormal price)

Bond Future Option

###  VarSwap

Variance Swap

## 8.Black Scholes

Listed Option, Listed Quanto, Warrant, Currency OTC

# VaR Calculation

Calculate PnL at the end of each VaR period (M=4days), use theoretical unrealized P&L of the position, calculated using the position’s fair value instead of market price, then perform N=5000 full model revaluation after each risk factor’s individual shock process to calculate post-shocked theoretical position P&Ls. Sorts N differences and pick x*N+1 -th worst one as VaR, 1-x is confidence level. 

$PnL_{sim}$ = Theoretical Value for position using shocked risk factors at time t – original theoretical value + cash flows from duting the time step t that are realized and included at t.

For each risk factors in a simulation, the risk factor will be shocked according to a specific shock process. 

Each risk factor has attributes: (Risk factor shock components)

-	Term vol: realized vool is calculated from historical time series for each specific risk factr
-	Random number based on 3 years of historical weekly return data, a correlatedd random number is calculated for each risk factr. Fallback logic is used
-	Drift based oon time horizon chosen, and given risk factor type or currency, a drift is included in shock process.

Two shock process: Lognormal shock process removes assumption of normality in the distribution of returns (profit loss), so resulting VaR incorporates asymmetrically distributed returns. While simulating random price path scenarios of an equity, the methodology follows GMB: 

- $dS_i = \mu_i S_i dt + \sigma_i S_i dz = drift + diffusion$

- $S_i$ = price of underlying risk factor I at time t

- dz = random variable in Wiener process, distributed normally with mean zero and variance dt 

- $\mu_i$= Instantaneous drift = expected rate of return of the underlying risk factor 

- $\sigma_i$ = Annualized vol of underlying (261 business days)
$Dt$ = specified time interval 

- $\mu_i, \sigma_i$ are const over time horizon, not a function of expiry. 

- $\mu_i$ is predetermined through finance curve of the underlying risk factor using risk-neutral interest rate. 

- $\sigma_i$ is extracted from term vool table of the underlying risk factor and can be adjusted manually.

- Diffusion term $\sigma_i S_i dz = \sigma_i \epsilon_i \sqrt{dt}$, $\epsilon_i$ ~$N$(0,1) is random variable drawn from a std unit normal distribution. 

Ito’s Lemma, this formula can be approximated by: 

$d\ln S_i = (\mu_i - \sigma_i^2 / 2) dt + \sigma_i \epsilon_i \sqrt{dt}$

The price path of underlying stock is simulatedd by sampling repeatedly for $\epsilon_i$ and substituting into the formula above. Each risk factor’s shock process is calculated independent of other risk factors, since random number Z provided is pre-calculate dwith correlatin preserved.

- $S_i^k = S_0^k \exp{(\mu_i - \sigma_i^2 /2)t + sigma_i Z_i^K \sqrt{t}}$

- $S_i^K$ = price of underlying risk factor i at time t for simulation k.

- $Z_i^K$ = random number of underlying risk factor i for simulation k

Given price of another underlying risk factor $S_j$, similar expression for shock amoount for each shock n:

$S_j^k = S_o^k e^{(\mu_j - \sigma_j^2 / 2)t + \sigma_j Z_j^K \sqrt{t}}$

Correlation of $Z_i^K$ and $Z_j^K$ is unbiased estimator of the correlation of $S_i, S_j$. Post-shock value is the multiplication of initial value before shock and shock amount. 

$S_j^k = S_0^k * shock_j^k$

## Normal Shock Process

Normal rather than lognormal. 

$dS_i = \mu_i dt + \sigma_i dz = drift + diffusion$

For lognormal process: 

$dS_i = \mu_i dt + \sigma_i \epsilon_i \sqrt{dt}$

$\mu_i, \sigma_i$ are const over time horizon, $\mu_i$ is predetermined through finance curve of underlying risk factor, using risk-neutral interest rate. $\sigma_i$ is extracted from term vol of underlying risk factor and can be adjusted manually. Ignore the mean since it's difficult to estimate and less important than diffusion.  Ito process for short horizons:

$dS_i = \sigma_i \epsilon_i \sqrt{dt}$

Price path of underlying stock is simulate by repeately sampling of $\epsilon_i$ and substituting in the formula. Each risk factor's shock process is calculated independent of other risk factors. 

$S_i^k = S_0^k + \sigma_i Z_i^k \sqrt{t}$

- $S_i^k$ = price of underlying risk factor i at time t for simulation k.

- $Z_i^k$ = random number of underlying risk factor i for simulation k 

- $S_j^k = S_0^k + \sigma_j Z_j^k \sqrt{t}$

Correlation of $Z_i^k$ and $Z_j^k$ is unbiased estimator of correlation of $S_i$ and $S_j$. So post shock value is sum of initial value before shock and shock amount.

$S_j^k = S_0^k + shock_j^k$

## Shock by Risk Factors

### Equities

Risk factors of Equities are shocked with lognormal shock process and include drift calculated from discount curve of associated default currency. 

Expected rate of return / drift for equity risk factor = risk-free rate of the curve associated with equity risk factor.

USD curve to determine risk-free rate: yield curve calculator  n-day rate for associatedd curve. Given shock process, term vol, risk-free rate, we can calculate post shocked value of equity risk factor. 

### Forward Curve (ROLL)
Every roll curve is generated from strip of futures ${\{F_1, ..., F_M}\}$, use lognormal forward curve process for ith futures contract Fi.

### Credit Spreads ( SR_Spread)

Credit Spread curves are shocked by shocking par spreads using lognormal shock process as calculated vols and correlations for log returns of 9 key tenors. Each tenor is shocked in a correlated fashion, just as for stock price returns.

### Vol (GAPVOL)

Vol at risk factors are shoocked dusing lognormal shock process. All tenors of individual GAPVOL risk factor share the same set of random numbers, with only term vool being distinct. This ensures 100% correlaton among vol shocks between expiry tenors and strikes.

### Currency (CURNCY)

Currency risk factor is shocked by using lognormal shock process, including drift calculated from corresponding discount curve of associated default currency.

### Interest Rates (FwdCrvSHIFT)

Interest rate shock may have either lognormal or normal diffusion. Adopting normal diffusion instead of lognormal diffusion, interest rate shock process for key rates simplifies. The lognormal/normal trigger for interest rate.

### Correlated Random Numbers Generation

Use canonical eigenvectors method by multiplying the matrix of normalized returns by a matrix of iid normal random numbers resulting in a matrix of correlated normal random numbers to be used in VaR. 3 years of correlation data using weekly returns = 155 eigenvectors.

__Algorithm__

- Obtain universe of risk factors (RF) of length N to represent shock coomponents for simulation.
- Generate a T by S # of simulations matrix Z of iid unit normal random numbers from a set seed in a Quasi Random Sobol Sequence which serves as a shared set of random numbers.
- Calculate correlated random numbers for each risk factor:
   - Obtain 3-year time series of weekly return data R of length T. Missing data is dealt with using the fall back logic, detailed in separate section.
   - Calculate std of weekly returns
   - Calculate set of correlated random numbers by multiplying the weekly return vector by the matrix of shared random variables.
   - Normalize random numbers by dividing weekly return vector by the matrix of shared random numbers.
   - Normalize random numbers by dividing by the std of weekly returns.

Combining each risk factor's vector of correlated random numbers, we create a N by S matrix of correlated random numbers.

$R_{N*T}*D*Z_{T*S} = RN_{N*S}$
- N = # risk factors
- T = length of 3-year weekly return 
- S = # simulations 
- D = Diagonal matrix whose diagonal corresponds to square root of quadratic variation of each risk factor 
- $R_{N*T}$ = N by T matrix of weekly returns, where row i is weekly return of the i-th risk factor
- $Z_{T*S} = T by S matrix of shared random numbers$
- $RN_{N*S}$ = N by S matrix of calculated correlated random numbers, where row i is correlated random number of the i-th risk factor.

# Pricing Models

MRM = IR Stripping engine that handles all PV, standard cases, vanilla IR, non-risky bond. It's divided into PV (simply discount payoff) and Standard. 

## Convertible Preferred Stock (CPS)

After last conversion date, the CPS becomes perpetual debt of the issuing company. If the CPS is not converte by the last conversion date, it's like a perpetual debt security. 

## FX Forward

Contracts for the delivery of a currency at s speficied rate on specified future date.

## Currency Future:

Contracts for purchase and delivery of currency at a specified price and time in future.

## Future (Equity Derivative)

Equity futures contracts are exchange-traded securities that are agreements to buy/sell and equity in future month at an agreed upon price.

## Quanto Future

A future with payout in a different currency from the underlying security currency, with a fixed exchange rate between the future's currency and the underlying security's currency. Payoff for quanto future = currency strike * underlying security price. 

## Preferred Stock

A security that shares characteristics of both stocks and bonds, has a fixed annual dividend or cooupon, but no maturity. The holder of preferred stock may or may not have privileges associated with common stock ownership.

# Fixed Income

## Floating Rate Note (FRN)

Flating rate note securities = floaters = bonds whose couopons change based on future interest rates, i.e. a bond with a variable rate coupon. FRNs' coupons are generally set as a spread above or below a specific benchmark rate. The interest paid adjusts based on changes in the reference rate. 

## Treasury Bond

It's a debt security issued by government, features includ: amortization, sinking notional, call/put provisions, etc.

## Treasury Bill

It's a short term debt instrument issued by US government, a T-Bill has a par value of $1000 and a maturity of less than one year. It's assumed to be free of credit risk.

# IR derivative

## Deliverable Swap Futures

US dollar denominated quarterly contracts expiring on IMM dates for key benchmark maturities (2,5,10,30years). At expiration, all open positions deliver into CME Group Cleared Interest Rate Swaps. 

## Bond Future

Bond Future is based on deliverable basket of bonds with widely different price and yield characteristics. As interest rate change, the different bonds from the basket can become cheapest to deliver bond. So the real futures contract behaves like a hybrid of bonds from a deliverable set weighted by their likelihood of being delivered. To model this, the deliverable bond index (DBIx) is the underlying of the bond futures contract.

# Standard

## Commodities

### Commodity Futures Spread

A future that trades on the difference in price between 2 reference indexes (2 separate points on the same curve or 2 points on different roll curves).

### Commodity Future

A futures contract that is an agreement to buy/sell a commodity or equity/index at a future date at an agreed upon price. Commodity future's price comes from its specified roll curve.

Model relies on normal diffusion rather than log-normal, but is identical to COF model.

## Currency

### American Depository Receipt (ADR)

A foreign security that is placed in US custodial account and traded on US market. 

### Fed Funds Futures (FFF)

Provide trading opportunities for mgmt of risk exposures associated with a variety of money market interest rates.

### Overnight Index Swap Future (OISF)

Similar to FFF, while FFF is based on average of overnight rate over the future's underlying contract period, OISF are based on geometric average (compounding) of OIS overnight rate (In US: Fed fund effective rate). The difference is encapsulated in the underlying indices (FFIx vs. OISx), not in futures themselves. 

### Forward Rate Agreement (FRA)

Forward contract on an interest rate. Traded on future level of 3- or 6-month LIBOR, but FRAs on any interest rate index are possible. Regular, basket, broken FRA payoffs.

# Black Model

Log-normal model of asset, should it be equity (Black Scholes) or commodity (Black), should it be adjusted (digits/barrier) or not.

## Commodity Future Options

Price come from roll curve affliated with commodity future.

## Convertible

### DECS (DCS/DCSp)

Convertible securities issued ina few different formats, all of which define a payoff vs. common share price graph characterized by 2 price hurdles, between which the return remains flat. DECS holder doesn't have option to convert prior to maturity, and conversion to ordinary shares at maturity is usually mandatory. The issuer may call DECS during time periods defined in the issue to force conversion. DCSp isn't new model, it just represents quoting convention, not a model change.

## Currency Derivative

Currency Future Option, Currency OTC Options

## Currency Exotic

### FX Barrier Option

Has been enhanced, now provides expanded monitoring options and a rebate setting.

### FX Digital Option (FXDO)

A new type of binary option where price at the time of expiry determines its binary status (whether or not out of position). It replaces the deprecating FXBI secutiry.

### FX Touch/Rebate Options (FXTO)

Similar to FXDO, FXTO security type is a new type of binary option. The distinguishing feature of FXTO is that there's no strike price. The security's binary status is solely dependent upon whether a barrier is hit or not, and therefore requires continuous monitorting. Oncee a barrier is touched, FXTO's binary status changes to out of position.

## Equity Derivative

### Future Option

### Forward Value Future Option

A future option whose premium is paid at expiration of the underlying future. Payoff for a forward value futures option (call) is $max[F-K,0]$.

### Basket Option

Option on equity basket underlying security. It entitles the holder to buy/sell a basket at particular strike price by/on expiration date. 

### Listed Option

Agreement to buy/sell an equity security at specific price and date. On the expiration date, holder can choose whether or not to exercise the right to buy/sell underlying equity at strike price.

### OTC Option

### Quanto Listed

Equity quant listed is an option whose payout is made in a different currency from that of the underlying security based on a fixed exchange rate. Option strike is expressed in the currency of the underlying equity, payoff for quant call option = $max[CurrencyStrike * (S-K),0]$

### Equity Right

A warrant, on a fractional number of shares.

### Warrant

An option that is issued by company. Underlying equity for a warrant is usually an index or a stock.

### Listed ADR Option

American depository receipt option is an option on an ADR  underlying security. Payoff for ADR call option is $max[S-K,0]$

## IR Derivative

### Bond Future Option

### Fed Fund Futures Option

## Whaley model

Variant of Black model dealing with American options. 

## Bachelier

### Commodity Future Spread Option

Assume the spread is treated as a normalprocess

## HazardRate

A determnistic intensity model, handles CDS and risky bonds. 

### Credit Default Swap (CDS)

Provides insurance against the default of a particular entity. Buyer of CDS makes regular payments to seller until either maturity of the contract or occurrence of a credit event (company defaults). If default by company, it's assumed the CDS buyer receives notional value of CDS from seller, while the defaulted reference obligation (bond/convertible bond) is transferred to the seller. CDS buyer also pays any portion of the payment that has accrued in the current period.

## Convertible

### Convertible Bond

May be converted to common shares of the issuer's stock or redeemed at par. Issuer may call the bond according to time periods and price. These call dates and values are entered into a call schedule. The convertible bond may/may not have a put schedule - a list of dates on which the bond may be sold back to the issuer. CNVI represents quoting convention.

### Exchangeable Convertible Bond (XCB)

Can be converted into common shares of a 3rd-party entity rather than common shares of the bond issuer. The issuer may/may not own this 3rd party's stock. Treatment of exchangeable bonds is different from standard convertibles in that: collateralization influences the value of an exchangeable convertible bond.

## Inflation

### Zero Coupon Inflation Swap (ZCIS)

Inflation derivative, an exchange of cash flows that allows reduce/increase your exposure to the risk of a decline in the purchasing power of money. An income stream that is tied to the ate of inflation is exchanged for an income stream with a fixed interest rate. But instead of acutally exchanging payments periodically, both income streams are paid as one lump-sum payment when the swap reaches maturity and the inflation level is known. 

The currency of the swap determines the price index used to calculate the rate of inflation. A swap denominated in $ wouold be based on the CPI of US. Other financial instruments that can be used to hedge against inflation risk are real yield inflation swaps, price index inflatin swaps, Treasury Inflation Protectedd Securities (TIPS), municipal and corporate inflation-linked securities, inflation-linked certificates of deposit, and inflation-linked savings bonds.

### Inflation Bond

Provides protection against inflation by linking its principal to an appropriately chosen inflation index (CPI).It's reset at the end of each coupon period to reflect the change in inflation over the period (for older inflation-linkedd 8-month lagged Gilts, the reset happens at the beginning of the period). The ratio between the end of the period index value and the value at the beginning of the period is used to compute inflation-adjusted principal. The inflation-adjusted principal is used to compute interest payment for this period. If inflation rate >0, principal amount on which the bond issuer will pay interest and final redemption will increase with time, protecting the bond holder from inflation-related loss.

## Hull White

Hull White 1-factor model handles interest rate options, bond options. 

## Swaption

European interest rate swaptions = options on interest rate swaps. Swaptions can be settled either by physical delivery or by cash settlement. If physical edlivery, the holder has the right to enter a contractually specified interest rate swap at a specific date in future at a fixed rate. 

### Money Market Future Option (MMFO)

Options on underlying money market future contract. Underlying of MMFO is a money market future (deposit futures). MMFO's can be American or European. Depending on the expiry of the option relative to the expiry of the underlying future, the MMFO's are either regular, when expiry of the option is the same as the expiry of the underlying future, or mid-curve options, in which the option expires significantly before the underlying futures contract.

### Caps and Floors

Caps (Floors) are a portfolio of simple interest rate options called caplets(floorlets). Each caplet (floorlet) is a European style call/put option on a single interest rate, like a money market rate LIBOR. Trading system has flexible cap and floor structures (CAPS) including editable schedules.

## Variance Swap

Variance swap uses static replication. It's a forward contract on the variance of the instrument's price over a period.

# Model Limitations

8 pricers (valuation models).

- Implied vol of commodity spread option is not shocked. 
- Whaley (and all American styles) model implementation for commodity averaging futures is not supported. Only European style options are supported.
- HazardRate Model doesn't account for bond subordination.
- Preferred Stock using PV model doesn't handle convertibility and perpetual maturities well. 
- Dividend futures don't follow a standard forward model as dividend changes are subject to discrete jumps. Need a new dividend model.
- Implication Model works well for equity and equity index options, but the implication of the vol surface with a parabolic fit currently can cause deep out of the money to be undervalued relative to the market - which may be mismarked due to stale prices. Improvements to the vol surface are progressing to handle pricing them.
- Implied vol are too low for equities that don't have options, but securities with optionality such as convertibles and warrants use the fallback the histotical volatility of the underlying. It'd be better to derive the vol directly than use the historical. 
- Currency Future Options (very small position count) uses historical vol of the currency pair or the surface from OTC options. The implie vols don't match the market and the pricing may be subject to errors.
- FX Forwards (very smalle position count) doesn't capture the market value. Impact is the FX risk exposure to the margin (portfolio) ccy is not captured. The impact is small for most clients as the risk is second order, but potitions with large market value may be under-risked.
- Fed Fund Options pricers cannot handle negative rates, now not an issue.
- Simplified Bond Future Option model is compared to Bond Future Model. CTD change risk isn't captured adquately.
- Basket Option cash leg is not correctly handled by the moddel when computing the implied vol. Very few numbers of positions are impacted.
- ATM Vol is used for Quanto Future adjustment while striked vol is used for Quanto option.
- Inflation: Seasonality adjustment was not included.

These limitations don't have a big impact on the price, and we don't expect to have any arbitrage strats playing on those second-order limitations. 

## Monte Carlo VaR

Second order issues:

- Model is not forward looking because VaR is calbrated with historical time series.
- Some risk factors could potentially partially capture risk (e.g. correlation and dividends aren't shocked)
- Normality assumption for all risk factors may in certain cases understate the risk
- By construction, Monte Carlo VaR can't capture extreme tail events easily.
- Hazard Rate diffusion doesn't account for the state of default. Only valuation changes due to default proobability changes. 
- Correlations are fixed using 3-year historical. Some risk may be missed due to changes in correlation regimes.
- Difference between theoretical price and actual price can cause VaR to be distorted. Especially evident in bond and equity options, where small differences can cause large differneces in starting point of the MC.
- Shocks on implied vol surfaces are nearly parallel across the whole surface since a sngle risk factor determines the correlation to the underlying price and other securities. It'd be better to have some twist in the shocks.
- Lack of vol surfaces and the use of historical vol of vols rather than tenorized vol for interest rate options leads to distorted vols and correlations.
- Monte Carlo VaR is designed for market risk. Any other type of risks (liquidity, concentration, operational) are not captured.

# Model Validation

Testing steps: pricing replication, VaR testing, external benchmarking, client portfolio sanity checks.

## Pricer Replication

99% of positions are validated through a full replication method. Aim of test is not only to check the pricers work exactly but also to better understand the pricers limitations.

## Methodology

$f_i$ = pricing function. Test is to compare both pricing functions with exact same inputs and validate the difference is close to 0.

Mass Pricing Validation. Because list of products validated through the full replication methods covers more than 99%.

## VaR Testing Steps

- Pricing Validation. Certify pricing works correctly.
- Calibration Inputs: Use pricers and pure market data to compute derived market data (vol surface, interest curve). We build a continuous market data universe out of discrete data point received from the market
- Risk Factor Selection: Among continuous universe build above, choose some discrete points as risk factors. They have time series and statistics attached to them, and allow us to diffuse the continuous universe.
- Risk Factor Time Series Construction: Certify how time series are built for any type of risk factor chosen above.
- Risk Factor Statistics computation
- Shock Calculation: Compute shocks applied to each risk factor using statistics above
- Diffusion of the Continuous Universe: How the continuous universe is rebuilt in a shock environment using shocks computed above from discrete risk factors
- MTF: Certify PnL used in VaR is computed using the same pricing function both on the initial continuous universe and post shock continuous universe computed above
- Aggregation: Final VaR at basic level is correctly computed by aggregating PnL of each position computed above.

## Calibration 

### Yield Curve Bootstrapping

Check that every yield curve used in our model is correctly calibrated (bootstrapped) based on market inputs, and exact same pricing function as the one validated in the pricing section.

- For the pricing of 1 type of security, we use a rate curves that we input ina specific pricing function to compute price. 
- To create rate curve, use bootstrapping method: use price of a strip of securities and a pricing function, but this time to estimate the rate curve that we should input into the model to match the market price. This shows output is a solution of inputs.
- $f_i$ = pricing function of strip i of interest rate validated in pricing
- $MarketInputs_{i,j}$ = j-th market inputs price of strip i retrieved froReuters
- $OutputCurve_i$ = output continuous curve resulting from bootstrapping of strip i
- $StripItec_{i,j} = f_i(OutputCurve_i, j)$ = price of interest rate instrument of type i with characteristics of j as recalculated by ITEC using pricing funcion validated in the pricing function.
- Check $\nabla_{i,j}StripItec_{i,j} = MarketInputs_{i,j}$

### Calibration based on market price

Check every calibrated market data are correctly calibrated based in markeet inputs and using exact same pricing function as validated. Reprice an instrument having the same characteristics as the market input using the output rates of the bootstrapper and check that is match market price using the same pricing function.

### Surface Equity I-VOL (Implied Vol)

- Volatility surface not marked by traders: For illiquid single name index the VOL isn't perfectly calibrated.
- Flat extrapolation of the tails: I-Vol sometimes too low on the tails
- Filtering logic on liquidity: to be enhanced 

### Calibration Inputs

YC Boostrap, Deposit Bootstrap, SWAP Bootstrap, Gov Bond Bootstrap, FRA bootstrap, FXForward Boostrap, Par Bond Strip Bootstrap, Zero Coupon Infla Boostrap, Equity Implied Vol, FX implied vol, Calibration of BDFO Vol, Hazard Curve, inflation curve, Calibration of Roll Yield, Calibration of Commo Vol, Calibration of IR Zero Spread, Calibration of BND CS Spread

## Risk Factor Time Series Construction

Equity/Commo/FX Time series, Roll Time series, YC Time series (par bonds), IVol time series, Time Series check/smoothing

### Simple Time Series

Time series made from past realized price. Trading system download prices from providers and inserts into time series. 

### Interpolated Time Series

Time series of generic objects derived from realized prices of market. 

- Example 1: From actual market swap prices we derived a continuous spot curve, but to avoid any rolling issue, we take generic tenors like 3 months, 1 year for time series. 

- Example 2: Vol surface, from actual option market prices, we derive a continuous implied vol surface and get generic tenor for time series (ATM30D, ATM60D)

### Smoothed Time Series

Linked to physical contract. On the day forward curve Risk Factor switch to rolling contract (next maturity) an artificial spike is created in time series. This spike impacts both correlation and historical time series. It has to be smoothed to better reflect the risk. 

## Risk Factors Statistics Computation

 Histo Vol Override, RNS proxy noise adjustment parameter, RNS time series proxy choce, histo vol commo proxy/ivol choice.

### Distribution Switch Test

 - Underlying Equity / FX: Lognormal
 - Forward Price: Lognormal
 - Vol: Lognormal
 - Interest rate: Lognormal/Normal
 - Credit: Lognormal

Challenge for interest rate: create a dynamic logic to switch from one distribution to the other depending on the level or the rates. Goal of the test is to check that this switching works as expected.

### Random Number Generation Test

Random numbers a $\mathcal{N}(0,1)$ random variables. There is one set of random numbers per risk factor. Random numbers are generated so that the correlation between each pair of risk factors is maintained during the diffusion. This is to keep "realistic" diffusion scenario in our Monte Carlo engine. The test is to check that random numbers are accurately compute. Trading system doesn't use standard variance-covariance matrix we have to test that those random numbers are unbiased estimator of 3-year weekly return correlation.

### Random Number FallBack Test

If risk factor doesn't have enough time series to compute a full 3 years of weekly returns, the missing returns are replaced by a set of simulated returns computed from a proxy returns and a random $\mathcal{N}(0,1)$ random variable.

- Normalized vector of simulated return used to backfill the missing return of the risk factor = $\rho$ * normalized vector of the proxy return over 3 years + $\sqrt{1-\rho^2} \mathcal{N}(0,1)$
- $\rho$  = the known correlation between risk factor and the proxy based on the available time returns

### Random Number Proxy Selection Test

Check trading system is applying correct proxy to compute the simulated returns above.

### Term Vol test

Historical volatility of each risk factr estimated based on a set of realized past returns. VaR takes 1-year of historical realized returns for each risk factor.

Annualized termvol for risk factor j based on n past returns $R_{j,i} = \ln (P_{j,i+1}/P_{j,i})$ (if distribution lognormal), $R_{j,i} =P_{j,i+1}-P_{j,i}$ (if distribution normal). $P_{i,j}$ = value of risk factor j at time i

$TVol_j = \sqrt{261*\Sigma_{i=0}^N (R_{j,i}^2)/N}$, volatility calculated in VaR computation

### Term Vol Fallback Test

Term vol uses 1 year historical return, if a risk factor doesn't have enough time series to compute full 1 year histrical returns, it falls back to term vol with less deepness.

## Shock Model Computation

The list of i-th shocks (for each MC simulation) for risk factor j computed, and used to diffuse pricing inputs in Monte Carlo engine.

### Volatility Override

For pegged currency risk factors, the time series is stale, and term vol is very low, might not capture future specific event. Be more conservative than simply using gross term 1y term vol. A transparent vol override has been design using several potential params to modify the vol in a more conservative way.

Reserve the lognormal model for each commo spread leg instead of normal model because the spread itself is more liquid in some cases. (Developing...)

Override Param = i-th param (cap/floor/multiplier) used to override the vol on entry table j 

Vol after override = vol recalculated after applying expected override for risk factor j

### Diffusion of continuous universe + Mark-to-model in a shocked world (MTF)

The two needs to be tested on the same time. Check mark-to-future computed by trading system = the one recomputed using same pricing function as the one validated in pricing validation. Use pricing inputs shocked by shocks computed with interpolation technique and time moving technique.

Mark-to-future of security i and simulation m moving 4 days forward as re-calculated, by using pricing function validated  + shock computed, + using interpolation technique and time moving technique.

Specify models of interpolation and techniques to "age" the position (mode the time forward). Example: how to handle a dividend or a coupon within the period of VaR. VaR uses 1 step move in time of 4 days, intead of 4 steps of 1 day to go to a 4 days horizon.

### Marginal / Incremental VaR

Specify the models of aggregation and technique to capture FX RISQ conversion

## Benchmarking

Shortcomings of risk engine model validation: 

- no credit risk factors
- can't handle negative interest rates
- less complex commodity models

## Risk Not in VaR Monitoring

Few risk factor not shocked in trading system:

- Quanto convexity (correlation between equity and FX not shocked)
- Smile vol (Smile distortion in strike not shocked)
- Basket Option (correlation between underlying not shocked)
- Dividend Risk (dividend used to compute forward price for single name is not shocked)

### Market Data Acuiqisition

When we require proxy, it means positions related to small exchanges don't distribute data, all or in some portion, through traditional data vendors like Reuters. The solution to the issue of proxies for pricing is to acquire these data sources directly, though IT cost are huge. This process was followed on ICE exchanges and reduced proxy count by 40000 positions. Trading system incorporate Nodal exchange, the next largest exchange not covered, and whose positions are all proxied. The remaining exchanges, which distribute some but not all their products, are EEX in Europe and Singapore Exchange. Most of these products on these exchanges are traded by corporate clients and are not a high proiority. 

# Glossary

- G**D: The Prime Services internal data warehouse
- P****d: The Prime Services internal software solution for the consolidation of counterparty risk monitoring and margin calls calculation
