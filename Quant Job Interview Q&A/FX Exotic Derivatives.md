FX Exotic Derivatives
====

Junfan Zhu

2021 - 06 - 08

<!-- TOC -->

- [FX Exotic Derivatives](#fx-exotic-derivatives)
- [1. FX Market Conventions](#1-fx-market-conventions)
  - [1.1. Domestic-Foreign Duality](#11-domestic-foreign-duality)
  - [1.2. Delta and At-the-money (ATM)](#12-delta-and-at-the-money-atm)
  - [1.3. FX Exotics Zoology](#13-fx-exotics-zoology)
  - [1.4. Valuation Formula for Exotics](#14-valuation-formula-for-exotics)
  - [1.5. Barrier Survival Probabilities for Brownian Motions](#15-barrier-survival-probabilities-for-brownian-motions)
  - [1.6. Valuations of Knock-Out Digitals](#16-valuations-of-knock-out-digitals)
  - [1.7. Barrier Survival Probabilities](#17-barrier-survival-probabilities)
  - [1.8. One-Touch Pay-at-Hit Options](#18-one-touch-pay-at-hit-options)
  - [1.9. Vanna-Volga Wizardry](#19-vanna-volga-wizardry)
- [2. Modeling](#2-modeling)
  - [2.1. Timeline](#21-timeline)
  - [2.2. Inputs](#22-inputs)
  - [2.3. Functions](#23-functions)
  - [2.4. Outputs](#24-outputs)
  - [2.5. Summary](#25-summary)
- [3. Options - Black Scholes Assumption](#3-options---black-scholes-assumption)
  - [3.1. Touch Options](#31-touch-options)
  - [3.2. Digital and Barrier Digital Options](#32-digital-and-barrier-digital-options)
  - [3.3. Barrier Options](#33-barrier-options)
  - [3.4. Domestic-Foreign Symmetry](#34-domestic-foreign-symmetry)
  - [3.5. Notation](#35-notation)
- [4. Barrier Digital Call Values](#4-barrier-digital-call-values)
  - [4.1. Full Monitoring](#41-full-monitoring)
  - [4.2. Partial-Start Monitoring](#42-partial-start-monitoring)
  - [4.3. Windows Monitoring](#43-windows-monitoring)
  - [4.4. European Monitoring](#44-european-monitoring)
  - [4.5. One-Touch Pay-at-Hit](#45-one-touch-pay-at-hit)
  - [4.6. Single Barrier](#46-single-barrier)
  - [4.7. Double Barrier](#47-double-barrier)
- [5. Vanna Volga Method for FX Exotics](#5-vanna-volga-method-for-fx-exotics)
- [6. Adjustments](#6-adjustments)
  - [6.1. Volatility Time Adjustments](#61-volatility-time-adjustments)
  - [6.2. Settle Adjustments](#62-settle-adjustments)
- [7. References](#7-references)

<!-- /TOC -->

Vanna-Volga method for exotic FX option pricing, can serve as a bias control and reduction for Monte Carlo simulations.

# 1. FX Market Conventions

A spot foreign exchange rate is quoted in market as price of one unit of *base currency* in *quote currency*. *Numeraire currency* is the currency under which the risk-neutral pricing measure is denominated, i.e. the *domestic currency*.

FX option, we need to pay attention to 2 additional currencies: *notional currency* and *premium currency*. Notional currency is either base or quote currency of the reference pair; it dominates the units of currency to be exchanged at a predetermined price (i.e. the strike, which is quoted in the same way as the spot) upon option exercise. Premium currency is trading currency of the option. It's normally the most frequently-traded currency of the pair.

Cash option price = premium quote $\times$ notional amount.

Digital option prices are quoted in percent of notionals.

- Foreign percent = Domestic pips / FX spot rate
- Domestic percent = Domestic pips / Strike
- Foreign pips = Domestic pips / Strike / FX spot rate

## 1.1. Domestic-Foreign Duality

An appropriately demoninated option value (commodity, dividend-yield paying stocks, etc.) shouldn't depend on choice of the numeraire, be it domestic or foreign. 

FX rate 

$$dX_t = (r_d - r_f) X_t dt + \sigma X_t dW_t$$

$\sigma$ = const vol, $r_d, r_f$ are domestic and foreign interest rates. The rate viewed from counterparty side is $\tilde{X_t} = 1/X_t$, by Ito's lemma

$$d\tilde{X}_t = (r_d - r_f) \tilde{X}_t dt + \sigma \tilde{X}_t d\tilde{W}_t$$

where $\tilde{W}_t = W_t - \sigma t$. By Girsanov theorem, $\tilde{W_t}$ is martingale under foreign measure $\mathbb{F}$, related to domestic measure $\mathbb{D}$ by Radon-Nikodym derivative

$$
\left(\frac{d \mathbb{F}}{d \mathbb{D}}\right)_{t}=e^{\sigma \widetilde{W}_{t}+\sigma^{2} t / 2}=\frac{X_{t}}{X_{0}} e^{-\left(r_{d}-r_{f}\right) t}
$$

So a call option satisfies

$$
E^{\mathbb{D}}\left[e^{-r_{d} T}\left(X_{T}-K\right)^{+}\right]=E^{\mathbb{F}}\left[\left.\frac{d \mathbb{D}}{d \mathbb{F}}\right|_{T} e^{-r_{d} T}\left(X_{T}-K\right)^{+}\right]=X_{0} K E^{\mathbb{F}}\left[e^{-r_{f} T}\left(\frac{1}{K}-\widetilde{X}_{T}\right)^{+}\right]
$$

So, under numeraire change, we need to interchange domestic and foreign rates, strike and its reciprocal, and call and put. We need to rescale the notional by the strike and convert the option premium currency by spot rate. Due to this dual relationship, we can rewrite the above formula as

$$
V(\phi)=X_{0} K V(\star \phi)
$$

where $\phi = \{\rm call, r_d, r_f, X_0, K\}, \rm and \star \phi = \{\rm put, r_f, r_d, 1/X_0, 1/K\}$ represent original and dual states.

## 1.2. Delta and At-the-money (ATM)

The sensitivity of an option price w.r.t. FX rate is option delta. For vanilla FX options, if premium currency is domestic currency,

- Spot deltas

$$
\Delta_{\text {call }}^{\mathrm{S}}=e^{-r_{f} T} \mathcal{N}\left(d_{+}\right), \quad \Delta_{\mathrm{put}}^{\mathrm{S}}=-e^{-r_{f} T} \mathcal{N}\left(-d_{+}\right)
$$

where 
$$
d_+ =\left(\ln F / K+\frac{1}{2} \sigma^{2} T\right) / \sigma \sqrt{T} $$

and forward is

$$ F=X_{0} e^{\left(r_{d}-r_{f}\right) T}
$$

- Forward deltas

$$
\Delta_{\text {call }}^{\mathrm{F}}=\mathcal{N}\left(d_{+}\right), \quad \Delta_{\text {put }}^{\mathrm{F}}=-\mathcal{N}\left(-d_{+}\right)
$$

---

If foreign currency is premium currency, the amount of foreign currency needs to be hedged and be reduced by the premium amount, so it's conventional to use premium-adjusted deltas.

- Premium-adjusted spot deltas

$$
\Delta_{\text {call }}^{\mathrm{S}, \mathrm{PA}}=\frac{K}{F} e^{-r_{f} T} \mathcal{N}\left(d_{-}\right), \quad \Delta_{\mathrm{put}}^{\mathrm{S}, \mathrm{PA}}=-\frac{K}{F} e^{-r_{f} T} \mathcal{N}\left(-d_{-}\right)
$$

where

$$
d_{-}=\left(\ln F / K-\frac{1}{2} \sigma^{2} T\right) / \sigma \sqrt{T}
$$

- Premium-adjusted forward deltas

$$
\Delta_{\text {call }}^{\mathrm{F}, \mathrm{PA}}=\frac{K}{F} \mathcal{N}\left(d_{-}\right), \quad \Delta_{\mathrm{put}}^{\mathrm{F}, \mathrm{PA}}=-\frac{K}{F} \mathcal{N}\left(-d_{-}\right)
$$

Premium-adjusted delta is the dual delta calculated in foreign measure (the amount of domestic currency needs to buy or sell), multiplied by $-1/X_0$ to convert to foreign currency. $\Delta_{\rm call}^{\rm S,PA}$ can be obtained from $\Delta_{\rm put}^S$ by interchanging $r_d$ and $r_f$, and rescaling the result by $-K/X_0$.

Vanilla FX options are specified in deltas rather than strikes. For spot, forward and premium-adjusted put deltas, we can invert normal cumulative distribution function $\mathcal{N}$ to find strike $K$. The premium-adjusted call delta is __not__ monotonic in strikes. To obtain unique solution, out-of-the-money or in-the-money flag should be specified. 

At-the-money (ATM) definitions in FX option:

- Spot ATM, $K_{\rm ATM} = X_0$
- Forward ATM, $K_{\rm ATM} = F$
- Delta-neutral ATM. Because ATM call and put have opposite deltas, so the strikes are $K_{\rm ATM} = Fe^{\frac{1}{2} \sigma^2 T}$ and $K_{\rm ATM}^{\rm PA} = Fe^{-\frac{1}{2} \sigma^2 T}$. Delta-neutral ATM deltas are

$$
\begin{array}{l}
\Delta_{\text {call }}^{\mathrm{S}}=-\Delta_{\mathrm{put}}^{\mathrm{S}}=\frac{1}{2} e^{-r_{f} T}, \Delta_{\text {call }}^{\mathrm{S}, \mathrm{PA}}=-\Delta_{\mathrm{put}}^{\mathrm{S}, \mathrm{PA}}=\frac{1}{2} e^{-r_{f} T-\frac{1}{2} \sigma^{2} T}, \Delta_{\text {call }}^{\mathrm{F}}=-\Delta_{\mathrm{put}}^{\mathrm{F}}=\frac{1}{2} \\
\Delta_{\text {call }}^{\mathrm{F}, \mathrm{PA}}=-\Delta_{\mathrm{put}}^{\mathrm{F}, \mathrm{PA}}=\frac{1}{2} e^{-\frac{1}{2} \sigma^{2} T}
\end{array}
$$

## 1.3. FX Exotics Zoology

One-touch (OT) option pays a predetermined amount, either in domestic currency (DC) or foreign currency (FC), if barriers are hit during the barrier monitoring window. Payoff can be paid at the maturity or at time of barrier breach.

No-touch (NT) option pays if the barriers are not hit. Payoff is paid at maturity, because whether barriers have been hit can only be made certain at maturity.

Knock-in (KI) barrier options pay the standard callor put payoff if barriers are hit during the barrier monitoring period. 

Knock-out (KO) options pay the payoff if the barriers are not hit. 
- Knock-out call (KOC).
- Up-and-out put (UOP).

They are all European, also carry rebate features. *Ex*: KI option can pay a certain amount of rebate if barriers aren't hit; the rebate is a no-touch option.

Barrier monitoring windows can be full windows (FW) covering the whole life of an option, or be partial, covering only first andlast portion of the option life, which is called partial-start (PS) or partial-end (PE) windows. If the option life span covers $0 \leq t \leq T$, barrier observations can start at $T_1$ and ends at $T_2$ with $0 \leq T_1 \leq T_2 \leq T$, which is called generic window (GW). 

- Full and partial-start windows = spot-starting windows (SSW)
- Partial-end and general windows = forward-starting windows (FSW).

The difference between the two is that there is one more twist for forward-starting windows. For knock-out options, when barriers are breached at start of the windows, the option can be simply knocked out (termed type I) or they can switch to options of a different kind (termed type II), e.g. from up-and-out to down-and-out option.

__How to deal with combinatorial explosion of exotics?__ Barrier option can be decomposed into digital options, $V_B = V_D^{FC} - KV_D^{DC}$ for a call, $V_B = KV_D^{DC} - V_D^{FC}$ for a put.

For touch options, paying unit domestic or foreign cash at maturity, the one-touch and no-touch option values sum up to PV of the cash payment.

$$
V_{\mathrm{OT}}+V_{\mathrm{NT}}=V_{\mathrm{cash}}=\left\{\begin{array}{ll}
e^{-r_{d} T} & (\mathrm{DC}) \\
e^{-r_{f} T} X_{0} & (\mathrm{FC})
\end{array}\right.
$$

Similarly, for digital options (KO = UO or DO for double barriers)

$$
\begin{array}{l}
V_{\mathrm{D}}^{\mathrm{KOC}}+V_{\mathrm{D}}^{\mathrm{KOP}}=V_{\mathrm{NT}}, \quad V_{\mathrm{D}}^{\mathrm{KIC}}+V_{\mathrm{D}}^{\mathrm{KIP}}=V_{\mathrm{OT}} \\
V_{\mathrm{D}}^{\mathrm{KOC}}+V_{\mathrm{D}}^{\mathrm{KIC}}=V_{\mathrm{D}}^{\text {call }}=\left\{\begin{array}{ll}
e^{-r_{d} T} \mathcal{N}\left(d_{-}\right) & (\mathrm{DC}) \\
e^{-r_{f} T} X_{0} \mathcal{N}\left(d_{+}\right) & (\mathrm{FC})
\end{array}\right. \\
V_{\mathrm{D}}^{\mathrm{KOP}}+V_{\mathrm{D}}^{\mathrm{KIP}}=V_{\mathrm{D}}^{\mathrm{put}}=\left\{\begin{array}{ll}
e^{-r_{d} T} \mathcal{N}\left(-d_{-}\right) & (\mathrm{DC}) \\
e^{-r_{f} T} X_{0} \mathcal{N}\left(-d_{+}\right) & (\mathrm{FC})
\end{array}\right.
\end{array}
$$

So we only need to calculate knock-out calls, $V_{\rm D}^{\rm UOC/DOC}$ for single barrier, and $V_D^{\rm KOC}$ for double barriers. The no-touch and one-touch option values can be recovered by taking limit $K \rightarrow
 0$. 

We compute the values of digital options with both domestic and foreign currency payoffs. By domestic-foreign duality, 

$$
V(\phi, \psi)=X_{0} V(\star \phi, \star \psi)
$$

where for a single barrier, $\phi = \{\rm DC, \rm up, B\}, \star \phi = \{\rm FC, \rm 1/L, 1/U\}$. This relation reduces the foreign payoff to the domestic payoff.

## 1.4. Valuation Formula for Exotics

By Girsanov theorem, we use change of variables, $x_t = \ln (X_t/ X_0) \sigma, dx_t = \mu dt + dW_t$, where $\mu = (r_d - r_f)/\sigma - \sigma / 2$. New variable $x_t = \mu t + W_t$ is a martingale under measure $\mathbb{P}$, related to measure $\mathbb{D}$ by

$$
\left(\frac{d \mathbb{P}}{d \mathbb{D}}\right)_{t}=\exp \left(-\mu x_{t}+\mu^{2} t / 2\right)
$$

Consider derivative contract with payoff $\mathcal{P}(\{X_i\})$ which depends on a set of $n$ observations $X_i = X(T_i)$ at time $T_i \in [0, T]$

Value of this derivative is 

$$
\begin{aligned}
V &=E^{\mathbb{D}}\left[e^{-r_{d} T} \mathcal{P}\left(\left\{X_{i}\right\}\right)\right]=E^{\mathbb{P}}\left[e^{-r_{d} T}\left(\frac{d \mathbb{D}}{d \mathbb{P}}\right)_{T} \mathcal{P}\left(\left\{X_{i}\right\}\right)\right] \\
&=e^{-r_{d} T} \int_{-\infty}^{\infty} d^{n} \bar{x}_{i} \phi_{n}\left(\left\{\bar{x}_{i}\right\} ; \Sigma\right) e^{\bar{\mu}_{n} \bar{x}_{n}-\frac{\mu_{n}^{2}}{2}} \mathcal{P}\left(\left\{X_{0} e^{\bar{\sigma}_{i} \bar{x}_{i}}\right\}\right)
\end{aligned}
$$

where $\bar{\mu}_i = \mu \sqrt{T_i}, \bar{\sigma}_i = \sigma \sqrt{T_i}, \bar{x}_i = x_i/ \sqrt{T_i}$, and values of standard Brownian motion $x_t$ under measure $\mathbb{P}$ at $T_i$ distribute as an $n$-dimensional normal $\phi_n$ with covariance $\mathrm{cov} (x_i, x_j) = \mathrm{min}(T_i, T_j)$. Normalizing the variables, the distribution has a correlation matrix $\Sigma$ whose non-diagonal entries are $\Sigma_{ij} = \Sigma_{ji} = \sqrt{T_i / T_j} = \rho_{ij}, i < j$.

If the payoff is analytic in $X_i$, the integration in above equation is of Gaussian type and can be easily derived. 

Fact about correlation matrix: $\rm det \Sigma = \Pi_{i = 1}^{n-1} (1-\rho_{i, i+1}^2)$, and its inverse is a symmetric tridiagonal matrix with entries $(\Sigma^{-1})_{11} = 1/(1-\rho_{12}^2), (\Sigma^{-1})_{ii} = 1/(1-\rho_{n-1, n}^2), (\Sigma^{-1})_{ii} = 1/(1-\rho_{i-1, i+1}^2)/[(1-\rho_{i-1,i}^2)(1-\rho_{i, i+1}^2)], (\Sigma^{-1})_{i, i+1} = (\Sigma^{-1})_{i+1, i} = -\rho_{i, i+1}/(1-\rho_{i, i+1}^2)$.

For digital options paying foreign cash at maturity, the option value is related to paying domestic cash by

$$
V_{\mathrm{D}, \mathrm{FC}}=X_{0} e^{\left(r_{d}-r_{f}\right) T} V_{\mathrm{D}, \mathrm{DC}}\left(\mu \rightarrow \mu^{\prime}\right)
$$

where 

$$
\mu^{\prime}=\mu+\sigma=\left(r_{d}-r_{f}\right) / \sigma+\sigma / 2
$$

The advantage of using $\mathbb{P}$-measure is that the FX rate process has been transformed to standard Brownian motion without drift.

## 1.5. Barrier Survival Probabilities for Brownian Motions

For standard Brownian motion $B_t$ without drift, the conditional survival probability for upper barrier $b$, by method of images,

$$
\operatorname{Pr}\left[\max _{0 \leq t \leq 1} B_{t} \leq b \mid x\right]=\frac{\phi(x)-\phi(\check{x})}{\phi(x)}=1-e^{-2 b(b-x)}, \quad b \geq x \geq 0
$$

where $\check{x} = 2b - x$ is a reflection.

Similarly, for down barrier,

$$
\operatorname{Pr}\left[\min _{0 \leq t \leq 1} B_{t} \geq b \mid x\right]=1-e^{-2 b(b-x)}, \quad b \leq x \leq 0
$$

The method of images can derive conditional probabilities for double barriers, 

$$
\left\{\max _{0 \leq t \leq 1} B_{t} \leq u\right\} \wedge\left\{\min _{0 \leq t \leq 1} B_{t} \geq \ell\right\}
$$

There are 2 sets of images, related to original change at $x$ and its reflected image $\check{x} = 2u -x$ by translations of $2nh$ where $h = u - l$. The conditional survival probability is

$$
\operatorname{Pr}\left[\left\{\max _{0 \leq t \leq t} B_{t} \leq u\right\} \wedge\left\{\min _{0 \leq t \leq 1} B_{t} \geq \ell\right\} \mid x\right]=\sum_{-\infty}^{\infty}\left[e^{-2 n h(n h-x)}-e^{-2(u+n h)(u+n h-x)}\right]
$$

The unconditional probability is the solution of a diffusion equation in a slab, 

$$
\partial_{t} p(t, x)=\frac{1}{2} \partial_{x}^{2} p(t, x)
$$

where the initial condition is $p(0, x)=\delta(x)$, and in addition $p(t,x)$ satisfies Dirichlet boundary conditions at $u$ and $l$. the solution is a Fourier series

$$
p(t, x)=\frac{2}{h} \sum_{n=1}^{\infty} e^{-\omega_{n}^{2} t / 2} \sin \left(\omega_{n} u\right) \sin \left(\omega_{n}(x-u)\right)
$$

where 

$$w_{n}=n \pi / h, n \in \mathbb{Z}$$

are eigenmodes. Equivalence of two expressions follows from Poisson summation formula.


## 1.6. Valuations of Knock-Out Digitals

We compute central quantities, knock-out digital call paying unit domestic currency. We start with generic barrier windows, $0 \leq T_1 \le T_2 \leq T$. Taking various limits gives us results for other types of monitoring windows.

Type I single barrier up-and-out digital call value is

$$
V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{GW}-\mathrm{I}}=\frac{e^{-r_{d} T-\bar{\mu}_{3}^{2} / 2}}{\sqrt{(2 \pi)^{3} \operatorname{det} \Sigma}} \int_{-\infty}^{\bar{b}_{1}} \int_{-\infty}^{\bar{b}_{2}} \int_{\bar{k}_{3}}^{\infty} d^{3} \bar{x}\left(e^{-\frac{1}{2} \bar{x}^{T} \Sigma^{-1} \bar{x}+\bar{x}^{T} u}-e^{-2 \bar{b}_{12}^{2}-\frac{1}{2} \bar{x}^{T} \Sigma_{1}^{-1} \bar{x}+\bar{x}^{T} v}\right)
$$

$$
=e^{-r_{d} T-\bar{\mu}_{3}^{2} / 2}\left[e^{\frac{1}{2} u^{T} \Sigma u} \int_{-\infty}^{\bar{b}_{1}-(\Sigma u)_{1}} \int_{-\infty}^{\bar{b}_{2}-(\Sigma u)_{2}} \int_{-\infty}^{-\bar{k}_{3}+(\Sigma u)_{3}} d^{3} \bar{x} \phi_{3}\left(\bar{x} ; \Sigma_{3}\right)\right.
$$

$$
\left.-e^{-2 \bar{b}_{12}^{2}+\frac{1}{2} v^{T} \Sigma_{1} v} \int_{-\infty}^{\bar{b}_{1}-\left(\Sigma_{1} v\right)_{1}} \int_{-\infty}^{\bar{b}_{2}-\left(\Sigma_{1} v\right)_{2}} \int_{-\infty}^{-\bar{k}_{3}+\left(\Sigma_{1} v\right)_{3}} d^{3} \bar{x} \phi_{3}\left(\bar{x} ; \Sigma_{2}\right)\right]
$$

$$
\begin{aligned}
=e^{-r_{d} T} &\left[\mathcal{N}_{3}\left(\bar{b}_{1}-\bar{\mu}_{1}, \bar{b}_{2}-\bar{\mu}_{2},-\bar{k}_{3}+\bar{\mu}_{3} ; \Sigma_{3}\right)\right.\\
-&\left.e^{2 \mu b} \mathcal{N}_{3}\left(\bar{b}_{1}+\bar{\mu}_{1},-\bar{b}_{2}-\bar{\mu}_{2}, \check{\bar{k}}_{3}+\bar{\mu}_{3} ; \Sigma_{2}\right)\right]
\end{aligned}
$$

where reflection is

$$
u=\left(0,0, \bar{\mu}_{3}\right)^{T}, v=\left(2 \rho_{12}^{2} \bar{b}_{1} /\left(1-\rho_{12}^{2}\right), 2 \bar{b}_{2} /\left(1-\rho_{12}^{2}\right), \bar{\mu}_{3}\right)^{T} . \check{\bar{k}}_{3}=2 \bar{b}_{3}-\bar{k}_{3}
$$

$\Sigma_{i}=J_{i} \Sigma J_{i}^{-1}$, with $J_i$ inversion matrix, $J_1 = \rm diag(-1,1,1)$, $\Sigma u=\left(\bar{\mu}_{1}, \bar{\mu}_{2}, \bar{\mu}_{3}\right)^{T}, \Sigma_{1} v=\left(-\bar{\mu}_{1}, 2 \bar{b}_{2}+\bar{\mu}_{2}, 2 \bar{b}_{3}+\bar{\mu}_{3}\right)^{T}, u^{T} \Sigma u=\bar{\mu}_{3}^{2}, v^{T} \Sigma_{1} v=4 \mu b+4 \bar{b}_{12}^{2}+\bar{\mu}_{3}^{2}$.

Type II option caries an additional piece from integration over the down-and-out call domain $\bar{x}_{1} \geq \bar{b}_{1}, \bar{x}_{2} \geq \bar{b}_{2}, \bar{x}_{3} \geq \bar{k}_{3}$,

$$
\begin{aligned}
V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{GW}-\mathrm{II}}=V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{GW}-\mathrm{I}}+e^{-r_{d} T}\left[\mathcal{N}_{3}\left(-\bar{b}_{1}+\bar{\mu}_{1},-\bar{b}_{2}+\bar{\mu}_{2},-\bar{k}_{3}+\bar{\mu}_{3} ; \Sigma\right)\right.
\end{aligned}
$$

$$
\left.-e^{2 \mu b} \mathcal{N}_{3}\left(-\bar{b}_{1}-\bar{\mu}_{1}, \bar{b}_{2}+\bar{\mu}_{2}, \check{\bar{k}}_{3}+\bar{\mu}_{3} ; \Sigma_{1}\right)\right]
$$

Double barrier is similar

$$
V_{\mathrm{KOC} / \mathrm{DC}}^{\mathrm{GW}}=e^{-r_{d} T} \sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}_{3}\left(\left[\bar{\ell}_{1}-\bar{\mu}_{1}, \bar{u}_{1}-\bar{\mu}_{1}\right],\left[\bar{\ell}_{2, n}-\bar{\mu}_{2}, \bar{u}_{2, n}-\bar{\mu}_{2}\right],-\bar{k}_{3, n}+\bar{\mu}_{3} ; \Sigma_{3}\right)\right.
$$

$$
\left.-e^{2 \mu u} \mathcal{N}_{3}\left(\left[\bar{\ell}_{1}+\bar{\mu}_{1}, \bar{u}_{1}+\bar{\mu}_{1}\right],\left[-\bar{\ell}_{2, n}-\bar{\mu}_{2},-\check{u}_{2, n}-\bar{\mu}_{2}\right], \overline{\bar{k}}_{3, n}+\bar{\mu}_{3} ; \Sigma_{2}\right)\right\}
$$

where $\bar{x}_{n}=\bar{x}-2 n \bar{h}$ for $x = u, l, k$, and the reflections are w.r.t. upper barrier, e.g., $\check{\bar{\ell}}_{2, n}=2 \bar{u}_{2}-\bar{\ell}_{2, n}$

Taking limit $T_1 \rightarrow 0$, and relabeling $T_2$ as $T_1$ and $T_3$ as $T_2$, we get fair values of a partial-start single-barrier digital option (where $\rho = \sqrt{T_1/T_2}$)

$$
V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{PS}}=e^{-r_{d} T}\left[\mathcal{N}_{2}\left(\bar{b}_{1}-\bar{\mu}_{1},-\bar{k}_{2}+\bar{\mu}_{2} ;-\rho\right)-e^{2 \mu b} \mathcal{N}_{2}\left(-\bar{b}_{1}-\bar{\mu}_{1}, \check{\bar{k}}_{2}+\bar{\mu}_{2} ;-\rho\right)\right]
$$

and its double barrier counterpart

$$
V_{\mathrm{KOC} / \mathrm{DC}}^{\mathrm{PS}}=e^{-r_{d} T} \sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}_{2}\left(\left[\bar{\ell}_{1, n}-\bar{\mu}_{1}, \bar{u}_{1, n}-\bar{\mu}_{1}\right],-\bar{k}_{2, n}+\bar{\mu}_{2} ;-\rho\right)\right. 
$$

$$
\left.-e^{2 \mu u} \mathcal{N}_{2}\left(\left[-\check{\bar{\ell}}_{1, n}-\bar{\mu}_{1},-\check{u}_{1, n}-\bar{\mu}_{1}\right], \check{\bar{k}}_{2, n}+\bar{\mu}_{2} ;-\rho\right)\right\}
$$

Similarly, the partial-end limits are

$$
V_{\mathrm{KOC} / \mathrm{DC}}^{\mathrm{PS}}=e^{-r_{d} T} \sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}_{2}\left(\left[\check{\bar{\ell}}_{1, n}-\bar{\mu}_{1}, \check{\bar{u}}_{1, n}-\bar{\mu}_{1}\right],-\bar{k}_{2, n}+\bar{\mu}_{2} ;-\rho\right)\right.
$$

$$
\left.-e^{2 \mu b} \mathcal{N}_{2}\left(\bar{b}_{1}+\bar{\mu}_{1},\left[-\check{\bar{k}}_{2}-\bar{\mu}_{2},-\bar{b}_{2}-\bar{\mu}_{2}\right] ;-\rho\right)\right]
$$

$$
V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{PE}-\mathrm{II}}=V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{PE}-\mathrm{I}}+e^{-r_{d} T}\left[\mathcal{N}_{2}\left(-\bar{b}_{1}+\bar{\mu}_{1},-\max \left(\bar{b}_{2}, \bar{k}_{2}\right)+\bar{\mu}_{2} ; \rho\right)\right.
$$

$$
\left.-e^{2 \mu b} \mathcal{N}_{2}\left(-\bar{b}_{1}-\bar{\mu}_{1}, \min \left(\bar{b}_{2}, \check{\bar{k}}_{2}\right)+\bar{\mu}_{2} ;-\rho\right)\right]
$$

$$
V_{\mathrm{KOC} / \mathrm{DC}}^{\mathrm{PE}}=e^{-r_{d} T} \theta(\bar{u}-\bar{k}) \times
$$

$$
\sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}_{2}\left(\left[\bar{\ell}_{1}-\bar{\mu}_{1}, \bar{u}_{1}-\bar{\mu}_{1}\right],\left[\max \left(\bar{\ell}_{2, n}, \bar{k}_{2, n}\right)-\bar{\mu}_{2}, \bar{u}_{2, n}-\bar{\mu}_{2}\right] ; \rho\right)\right. 
$$

$$
\left.-e^{2 \mu u} \mathcal{N}_{2}\left(\left[\bar{\ell}_{1}+\bar{\mu}_{1}, \bar{u}_{1}+\bar{\mu}_{1}\right],\left[-\min \left(\check{\bar{\ell}}_{2, n}, \check{\bar{k}}_{2, n}\right)-\bar{\mu}_{2},-\check{\bar{u}}_{2, n}-\bar{\mu}_{2}\right],-\rho\right)\right\}
$$

where $\theta(x)$ is Heaviside function.

Taking limit again, we gain full windows

$$
V_{\mathrm{UOC} / \mathrm{DC}}^{\mathrm{FW}}=e^{-r_{d} T} \theta(\bar{b}-\bar{k})\left[\mathcal{N}([\bar{k}-\bar{\mu}, \bar{b}-\bar{\mu}])-e^{2 \mu b} \mathcal{N}([-\check{\bar{k}}-\bar{\mu},-\bar{b}-\bar{\mu}])\right]
$$

$$
V_{\mathrm{KOC} / \mathrm{DC}}^{\mathrm{FW}}=e^{-r_{d} T} \theta(\bar{u}-\bar{k}) \sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}\left(\left[\max \left(\bar{\ell}_{n}, \bar{k}_{n}\right)-\bar{\mu}, \bar{u}_{n}-\bar{\mu}\right]\right)\right.
$$

$$
\left.-e^{2 \mu u} \mathcal{N}\left(\left[-\min \left(\check{\bar{\ell}}_{n}, \check{\bar{k}}_{n}\right)-\bar{\mu},-\check{\bar{u}}_{n}-\bar{\mu}\right]\right)\right\}
$$

So we get the well-known single-barrier up-and-out call option value.

$$
V_{\mathrm{B}, \mathrm{UOC}}=\theta(B-K)\left\{X_{0} e^{-r_{f} T}\left[\mathcal{N}\left(\left[d_{+}^{X_{0}}(B), d_{+}^{X_{0}}(K)\right]\right)-\left(\frac{B}{X_{0}}\right)^{2 \mu / \sigma} \mathcal{N}\left(\left[d_{+}^{\check{X}_{0}}(B), d_{+}^{\check{X}_{0}}(K)\right]\right)\right]\right.
$$

$$
\left.-K e^{-r_{d} T}\left[\mathcal{N}\left(\left[d_{-}^{X_{0}}(B), d_{-}^{X_{0}}(K)\right]\right)-\left(\frac{B}{X_{0}}\right)^{2 \mu / \sigma} \mathcal{N}\left(\left[d_{-}^{\check{X}_{0}}(B), d_{-}^{\check{X}_{0}}(K)\right]\right)\right]\right\}
$$

where 

$$
\check{X}_{0}=B^{2} / X_{0} \text { and } d_{\pm}^{x}(y)=\ln \left(x e^{\left(r_{d}-r_{f}\right) T} / y\right) / \sigma \sqrt{T} \pm \sigma \sqrt{T} / 2
$$

The infinite sums in the equations of double barriers can be numerically approximated by finite sums. Since larger $n$ correspond to more distant images, the series is fast converging.

## 1.7. Barrier Survival Probabilities

Barrier survival probabilities are recovered from equations above under limit $K \rightarrow 0$ and by dropping the overall discount factors. For spot-starting windows ending at $T$, we have

$$
\operatorname{Pr}_{\mathrm{SB}}^{\mathrm{SSW}}(\tau>T)=\mathcal{N}(s(\bar{b}-\bar{\mu}))-e^{2 \mu b} \mathcal{N}(-s(\bar{b}+\bar{\mu}))
$$

$$
\operatorname{Pr}_{\mathrm{DB}}^{\mathrm{SSW}}(\tau>T)=\sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left[\mathcal{N}\left(\left[\bar{\ell}_{n}-\bar{\mu}, \bar{u}_{n}-\bar{\mu}\right]\right)-e^{2 \mu u} \mathcal{N}\left(\left[-\check{\bar{\ell}}_{n}-\bar{\mu},-\breve{\bar{u}}_{n}-\bar{\mu}\right]\right)\right]
$$

where $\tau$ is the first time a stochastic FX rate hits barriers, *first-passage time*.

For a single barrier, $\tau = \inf(t; X_t \geq B)$ for up touch, and $\tau = \inf(t; X_t \leq B)$ for down touch.

For double barriers, $\tau = \rm min(\tau_U, \tau_L)$ where $\tau_U$ and $\tau_L$ are the first-passage times for the upper and lower barriers. 
Sign $s = \rm sgn(\bar{b}), +, (-)$ for an up (down) barrier.

For forward-starting windows that start at $T_1$ and end at $T_2$, probabilities are

$$
\operatorname{Pr}_{\mathrm{SB}}^{\mathrm{FSW}}(\tau>T)=\mathcal{N}_{2}\left(s\left(\bar{b}_{1}-\bar{\mu}_{1}\right), s\left(\bar{b}_{2}-\bar{\mu}_{2}\right) ; \rho\right)-e^{2 \mu b} \mathcal{N}_{2}\left(s\left(\bar{b}_{1}+\bar{\mu}_{1}\right),-s\left(\bar{b}_{2}+\bar{\mu}_{2}\right) ;-\rho\right)
$$

$$
\operatorname{Pr}_{\mathrm{DB}}^{\mathrm{FSW}}(\tau>T)=\sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left\{\mathcal{N}_{2}\left(\left[\bar{\ell}_{1}-\bar{\mu}_{1}, \bar{u}_{1}-\bar{\mu}_{1}\right],\left[\bar{\ell}_{2, n}-\bar{\mu}_{2}, \bar{u}_{2, n}-\bar{\mu}_{2}\right] ; \rho\right)\right.
$$

$$
\left.-e^{2 \mu u} \mathcal{N}_{2}\left(\left[\bar{\ell}_{1}+\bar{\mu}_{1}, \bar{u}_{1}+\bar{\mu}_{1}\right],\left[-\check{\bar{\ell}}_{2, n}-\bar{\mu}_{2},-\bar{u}_{2, n}-\check{\bar{\mu}}_{2}\right],-\rho\right)\right\}
$$

## 1.8. One-Touch Pay-at-Hit Options

One-touch pay-at-hit option values depend on the distribution of the first passage time. First consider a single barrier. The survival probability is computed in previous section. The distribution density for the first passage-time follows an inverse Gaussian distribution

$$
f_{\mathrm{SB}}^{\mathrm{SSW}}(\tau)=\frac{|b|}{\tau^{3 / 2}} \phi\left(\frac{b}{\sqrt{\tau}}-\mu \sqrt{\tau}\right)
$$

So the one-touch option paying unit domestic currency at hit is

$$
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{SB}, \mathrm{SSW}}=\int_{0}^{T} f_{\mathrm{SB}}^{\mathrm{SSW}}(\tau) e^{-r_{d} \tau} d \tau=e^{(\mu-\alpha) b}\left[\mathcal{N}(s(-\bar{b}+\bar{\alpha}))+e^{2 \alpha b} \mathcal{N}(-s(\bar{b}+\bar{\alpha}))\right]
$$

where 

$$
\alpha=\sqrt{\mu^{2}+2 r_{d}}, \bar{\alpha}=\alpha \sqrt{T}
$$

Values for one-touch options paying unit foreign currency are 

$$
V_{\mathrm{OT}, \mathrm{FC}}^{\mathrm{SB}}=B V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{SB}}
$$

---

Spot-starting double window survival probability is also computed in previous section, the corresponding distribution density function is

$$
f_{\mathrm{DB}}^{\mathrm{SSW}}(\tau)=\frac{1}{\tau} \sum_{n=-\infty}^{\infty} e^{2 \mu n h}\left[\phi\left(\bar{u}_{n, \tau}-\bar{\mu}_{\tau}\right) \bar{u}_{n, \tau}-\phi\left(\bar{\ell}_{n, \tau}-\bar{\mu}_{\tau}\right) \bar{\ell}_{n, \tau}\right]
$$

Partial sums of the above equation are slowly converging, so we need to write it in the dual form in momentum space,

$$
f_{\mathrm{DB}}^{\mathrm{SSW}}(\tau)=\sum_{n=1}^{\infty} \frac{\lambda_{n} \omega_{n}}{h} e^{-\left(\omega_{n}^{2}+\mu^{2}\right) \tau / 2} \sin \left(\omega_{n} u\right)
$$

where $\lambda_{n}=e^{\mu u}-(-)^{n} e^{\mu \ell}$. The resulting series has a much better converging property because high frequency terms are exponentially suppressed.

The one-touch value, by direct integration, is

$$
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{DB}, \mathrm{SSW}}=\int_{0}^{T} f_{\mathrm{DB}}^{\mathrm{SSW}}(\tau) e^{-r_{d} \tau} d \tau=\sum_{n=1}^{\infty} \frac{2 \lambda_{n} \omega_{n}}{\left(\omega_{n}^{2}+\alpha^{2}\right) h}\left(1-e^{-\left(\omega_{n}^{2}+\alpha^{2}\right) T / 2}\right) \sin \left(\omega_{n} u\right)
$$

$$
=p_{U} e^{(\mu-\alpha) u}+p_{L} e^{(\mu-\alpha) \ell}
$$

where

$$
p_{U}=p-S(u), p_{L}=q+S(\ell), p=\left(e^{2 \alpha h}-e^{2 \alpha u}\right) /\left(e^{2 \alpha h}-1\right), q=1-p
$$

$$
S(x)=\sum_{n=-\infty}^{\infty} \frac{\omega_{n} \sin \left(\omega_{n} x\right)}{\left(\omega_{n}^{2}+\alpha^{2}\right) h} e^{-\left(\omega_{n}^{2}+\alpha^{2}\right) T / 2+\alpha x}
$$

We have applied Poisson summation formula to the non-exponential terms and used the residue theorem to compute Fourier transforms. $S(u) > 0, S(l) < 0, S(u)-S(\ell)=\operatorname{Pr}_{\mathrm{DB}, \alpha}^{\mathrm{SSW}}(\tau>T)$, survival probability of drift $\alpha$.

To price double-barrier one-touch option that pays the foreign currency, we follow domestic-foreign duality to replace $u \leftrightarrow-\ell, \mu \rightarrow-\mu^{\prime}, r_{d} \leftrightarrow r_{f}$, where $\alpha, h$ are invariant under these replacements.

$$
V_{\mathrm{OT}, \mathrm{FC}}^{\mathrm{DB}, \mathrm{SSW}}=p_{U} U e^{(\mu-\alpha) u}+p_{L} L e^{(\mu-\alpha) \ell}
$$

Consider up touch, by measure-changing gymnastics

$$
\begin{aligned}
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{SB}, \mathrm{SSW}} &=E^{\mathbb{D}}\left[e^{-r_{d} \tau} \mathbb{I}(\tau \leq T)\right]=E^{\mathbb{P}}\left[e^{\mu b-\mu^{2} \tau / 2-r_{d} \tau} \mathbb{I}(\tau \leq T)\right]=e^{(\mu-\alpha) b} E^{\mathbb{Q}}[\mathbb{I}(\tau \leq T)] \\
&=e^{(\mu-\alpha) b}\left[\mathcal{N}(-\bar{b}+\bar{\alpha})+e^{2 \alpha b} \mathcal{N}(-\bar{b}-\bar{\alpha})\right]
\end{aligned}
$$

where $\mathbb{Q}$ is measure under which the underlying process has a drift $\alpha$.

Similar results hold for double-barrier one-touch options. If an option pays fixed amount $P(Q)$ at upper (lower) barrier, then it has value 

$$
V_{\mathrm{OT}}^{\mathrm{DB}}=p_{U} P e^{(\mu-\alpha) u}+p_{L} Q e^{(\mu-\alpha) \ell}
$$

where

$$
p_{U}=\operatorname{Pr}\left(\tau_{U} \leq \tau_{L}, \tau_{U}<T\right), p_{L}=\operatorname{Pr}\left(\tau_{L} \leq \tau_{U}, \tau_{L}<T\right)
$$

Let $p_U = p - A(u), p_L = q + B(l)$, under limit $T \rightarrow \infty$, by Doob's __optional sampling theorem__, since $e^{-2\alpha B_t}$ is martingale for Brownian motion $B_t$ with drift $\alpha, p e^{-2 \alpha u}+q e^{-2 \alpha \ell}=1$. 

Obviously $A(u)-B(\ell)=\operatorname{Pr}_{\mathrm{DB}}^{\mathrm{SSW}}(T, u, \ell)$. By domestic-foreign duality, we have $B(l) = -e^{2 \alpha \ell} A(-\ell) \text { and } A(u)=-e^{2 \alpha u} B(-u)$, e.g. a derivative that pays unit domestic cash only if $\tau_U < \tau_L$. So we have 

$$
A(u)+e^{2 \alpha \ell} A(-\ell)=\operatorname{Pr}_{\mathrm{DB}}^{\mathrm{SSW}}(T, u, \ell)
$$

with solution

$$
A(u)=\left[\operatorname{Pr}_{\mathrm{DB}}^{\mathrm{SSW}}(T, u, \ell)-e^{2 \alpha \ell} \operatorname{Pr}_{\mathrm{DB}}^{\mathrm{SSW}}(T,-\ell,-u)\right] /\left(1-e^{-2 \alpha h}\right)=S(u), B(\ell)=S(\ell)
$$

The three times $\tau_U, \tau_L, T$ can be arranged in 6 ways, probabilities can be determined $(\tau = \rm min(\tau_U, \tau_L))$.

$$
\begin{array}{l}
\operatorname{Pr}\left(\tau_{L}<\tau_{U}<T\right)=\operatorname{Pr}\left(\tau_{U}<T\right)-p_{U} \\
\operatorname{Pr}\left(\tau_{U}<\tau_{L}<T\right)=\operatorname{Pr}\left(\tau_{L}<T\right)-p_{L} \\
\operatorname{Pr}\left(\tau_{U}<T<\tau_{L}\right)=\operatorname{Pr}(\tau<T)-\operatorname{Pr}\left(\tau_{L}<T\right) \\
\operatorname{Pr}\left(\tau_{L}<T<\tau_{U}\right)=\operatorname{Pr}(\tau<T)-\operatorname{Pr}\left(\tau_{U}<T\right) \\
\operatorname{Pr}\left(T<\tau_{U}<\tau_{L}\right)=S(u), \quad \operatorname{Pr}\left(T<\tau_{L}<\tau_{U}\right)=-S(\ell)
\end{array}
$$

All RHS are known, the probabilities can be used to price more exotic options, like knock-in-knock-out (KIKO) options.

For forward-starting windows, one-touch option values can be computed by conditioning,

$$
\begin{aligned}
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{SB}, \mathrm{FSW}-\mathrm{I}}=& e^{-r_{d} T_{1}} \mathcal{N}\left(s\left(-\bar{b}_{1}+\bar{\mu}_{1}\right)\right)+e^{(\mu-\alpha) b}\left[\mathcal{N}_{2}\left(s\left(\bar{b}_{1}-\bar{\alpha}_{1}\right), s\left(-\bar{b}_{2}+\bar{\alpha}_{2}\right) ;-\rho\right)\right.\\
&\left.+e^{2 \alpha b} \mathcal{N}_{2}\left(s\left(\bar{b}_{1}+\bar{\alpha}_{1}\right),-s\left(\bar{b}_{2}+\bar{\alpha}_{2}\right) ;-\rho\right)\right] \\
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{SB}, \mathrm{FSW}-\mathrm{II}}=& e^{(\mu-\alpha) b}\left\{\left[\mathcal{N}_{2}\left(\bar{b}_{1}-\bar{\alpha}_{1},-\bar{b}_{2}+\bar{\alpha}_{2} ;-\rho\right)+\mathcal{N}_{2}\left(-\bar{b}_{1}+\bar{\alpha}_{1}, \bar{b}_{2}-\bar{\alpha}_{2} ;-\rho\right)\right]\right.\\
&\left.+e^{2 \alpha b}\left[\mathcal{N}_{2}\left(\bar{b}_{1}+\bar{\alpha}_{1},-\bar{b}_{2}-\bar{\alpha}_{2} ;-\rho\right)+\mathcal{N}_{2}\left(-\bar{b}_{1}-\bar{\alpha}_{1}, \bar{b}_{2}+\bar{\alpha}_{2} ;-\rho\right)\right]\right\} \\
V_{\mathrm{OT}, \mathrm{DC}}^{\mathrm{DB}, \mathrm{FSW}}=& e^{-r_{d} T_{1}}\left(\mathcal{N}\left(\bar{\ell}_{1}-\bar{\mu}_{1}\right)+\mathcal{N}\left(-\bar{u}_{1}+\bar{\mu}_{1}\right)\right)+p_{U}^{\prime} e^{(\mu-\alpha) u}+p_{L}^{\prime} e^{(\mu-\alpha) \ell} \\
V_{\mathrm{OT.FC}}^{\mathrm{DB}, \mathrm{FSW}}=& e^{-r_{f} T_{1}} X_{0}\left(\mathcal{N}\left(\bar{\ell}_{1}-\bar{\mu}_{1}^{\prime}\right)+\mathcal{N}\left(-\bar{u}_{1}+\bar{\mu}_{1}^{\prime}\right)\right)+p_{U}^{\prime} U e^{(\mu-\alpha) u}+p_{L}^{\prime} L e^{(\mu-\alpha) \ell}
\end{aligned}
$$

where for double barriers $p_{U}^{\prime}=p^{\prime}-S^{\prime}(u), p_{L}^{\prime}=q^{\prime}+S^{\prime}(\ell)$,

$$
p^{\prime}=\left[e^{2 \alpha h} \mathcal{N}\left(\left[\bar{\ell}_{1}+\bar{\alpha}_{1}, \bar{u}_{1}+\bar{\alpha}_{1}\right]\right)-e^{2 \alpha u} \mathcal{N}\left(\left[\bar{\ell}_{1}-\bar{\alpha}_{1}, \bar{u}_{1}-\bar{\alpha}_{1}\right]\right)\right] /\left(e^{2 \alpha h}-1\right)
$$

$q' = 1-p'$,

$$
I_{n}(x)=\frac{e^{\omega_{n}^{2} T_{1} / 2}}{\sqrt{2 \pi T_{1}}} \int_{0}^{h} d z e^{-(z-x)^{2} / 2 T_{1}} \sin \left(\omega_{n} z\right)
$$

and

$$
S^{\prime}(x)=\sum_{n=-\infty}^{\infty} \frac{\omega_{n} I_{n}(x)}{\left(\omega_{n}^{2}+\alpha^{2}\right) h} e^{-\left(\omega_{n}^{2}+\alpha^{2}\right) T_{2} / 2+\alpha x}
$$

Type I options cease to exist, when barriers are breached out at the start of monitoring, but type II options turn to options of a different kind.

## 1.9. Vanna-Volga Wizardry

We assumed const FX rate volatilities above. Now we assume volatility smiles (or commonly translate vanilla prices to implied vol). Although not self-consistent and arbitrage free, Vanna-Volga method is popular among practitioners to price the first generation exotics.

Vanna-Volga attempts to hedge against implied volatility movements through vega ($\mathcal{V} = \partial V / \partial \sigma$), vanna ($\mathcal{V}_1 = \partial^2V / \partial \sigma \partial X$), volga ($\mathcal{V}_2 = \partial^2 V / \partial \sigma^2$), so any derivative can be priced as 

$$
V_{\rm Vanna-Volga} = V_0 + \mathcal{V} C_\mathcal{V} + \mathcal{V}_1 C_{\mathcal{V}_1} + \mathcal{V}_2 C_{\mathcal{V}_2}
$$

where $V_0$ is fair value with a const ATM volatility.

The overhedge cost $C_\mathcal{V}, C_{\mathcal{V}_1}, C_{\mathcal{V}_2}$ can be found by matching the Vanna-Volga prices of three most liquid options with their market, e.g., ATM, $25\Delta$ risk-reversal (RR) and $25 \Delta$ butterfly (BF).

$$
\left(\begin{array}{c}
V_{\mathrm{MKT}}^{\mathrm{ATM}} \\
V_{\mathrm{MKT}}^{\mathrm{RR}} \\
V_{\mathrm{MKT}}^{\mathrm{BF}}
\end{array}\right)=\left(\begin{array}{c}
V_{0}^{\mathrm{ATM}} \\
V_{0}^{\mathrm{RR}} \\
V_{0}^{\mathrm{BF}}
\end{array}\right)+\left(\begin{array}{ccc}
\mathcal{V}^{\mathrm{ATM}} & \mathcal{V}_{1}^{\mathrm{ATM}} & \mathcal{V}_{2}^{\mathrm{ATM}} \\
\mathcal{V}^{\mathrm{RR}} & \mathcal{V}_{1}^{\mathrm{RR}} & \mathcal{V}_{2}^{\mathrm{RR}} \\
\mathcal{V}^{\mathrm{BF}} & \mathcal{V}_{1}^{\mathrm{BF}} & \mathcal{V}_{2}^{\mathrm{BF}}
\end{array}\right)\left(\begin{array}{c}
C_{\mathcal{V}} \\
C_{\mathcal{V}_{1}} \\
C_{\mathcal{V}_{2}}
\end{array}\right)
$$

The equation can be inverted to find the $C$'s.

To price barrier options, the Vanna-Volga formula is modified further by fudge factors 

$$
V_{\text {Vanna-Volga }}=V_{0}+p_{\mathcal{V}} \mathcal{V} C_{\mathcal{V}}+p_{\mathcal{V}_{1}} \mathcal{V}_{1} C_{\mathcal{V}_{1}}+p_{\mathcal{V}_{2}} \mathcal{V}_{2} C_{\mathcal{V}_{2}}
$$

For knock-out options, these factors are usually the barrier survival probabilities. 

Vanna-Volga price of a vanilla option of strike $K$ can be written as a combination of 3 pivot option prices

$$
V_{\text {Vanna-Volga }}(K)=V_{\mathrm{BS}}(K)+\sum_{i=1}^{3} w_{i}(K)\left(V_{\mathrm{MKT}}\left(K_{i}\right)-V_{\mathrm{BS}}\left(K_{i}\right)\right)
$$

where 

$$
w_{i}(K)=\frac{\mathcal{V}(K)}{\mathcal{V}\left(K_{i}\right)} \prod_{j=1, j \neq i}^{3} \frac{\ln K_{j} / K}{\ln K_{j} / K_{i}}
$$

Note that $w_i(K_j) = \delta_{ij}$, so the above is reminiscent of the Lagrange interpolation. Inverting the Vanna-Volga prices to volatilities can be used as an interpolation of implied volatilities.

----

# 2. Modeling

## 2.1. Timeline

$$t_0, t_{S_u}, t_\epsilon, t_{\epsilon + S_u}, t_{\epsilon + S_x}$$

## 2.2. Inputs

- $t_0$ = Trade date

- $t_\epsilon$ = Option expiry time

- $t_{S_u}$ = Underlying settle time

- $t_{e + S_u}$ = Option expiry time + underlying settle time 

- $t_{e + S_x}$ = Option expiry time + exercise settle time

- $\tau(t_i, t_j)$ = Interest time between time $t_i$ and $t_j$ in years, according to some day count convention

- $\tau^V(t_i, t_j)$ = Volatility time between ...

- $X_0$ = FX spot rate

- $r_d (t_i, t_j)$ = Continuously compounded domestic interest rate between $t_i$ and $t_j$. $t_d$ = $r_d (t_0, t_e)$.

- $r_f (t_i, t_j)$ = Continuously compounded foreigh interest rate between $t_i$ and $t_j$. $t_f = r_d(t_0, t_e)$.

- $\sigma$ = Const volatility

- $B$ = Single barrier level 
- $U$ = Double barrier upper levels
- $L$ = Double barrier lower levels
- $K$ = Strike of the option


## 2.3. Functions

- $\Phi(x)$ = Standard normal CDF (1-dimensional)
- $\Phi(x; \Sigma)$ = Standard normal CDF (n-dimensional)
- $\phi(x)$ = standard normal PDF (1-dimensional)
- $\phi_n(x; \Sigma)$ = Standard normal PDF (n-dimensional)
- $X$ = Adjusted FX spot rate (for settlement)
- $\Sigma$ = Correlation matrix

## 2.4. Outputs

- $V$ = Fair value of exotics (Black-Scholes)
- $V_{Vanna~Volga}$ = Fair value of exotics (Vanna Volga)
- $V'$ = Fair value of exotics after settle adjustment

## 2.5. Summary


- $X_{t}:$ FX rate at time $t, X_{0}:$ FX spot rate
- $r_{d}:$ domestic interest rate, $r_{f}:$ foreign interest rate
- $\sigma$ : constant volatility
- $\mu=\left(r_{d}-r_{f}\right) / \sigma-\sigma / 2, \mu^{\prime}=\left(r_{d}-r_{f}\right) / \sigma+\sigma / 2, \alpha=\sqrt{\mu^{2}+2 r_{d}}$
- $\bar{\mu}=\mu \sqrt{T}, \bar{\sigma}=\sigma \sqrt{T}, \bar{\alpha}=\alpha \sqrt{T}$
- $B:$ single barrier level, $U, L:$ double barrier upper and lower levels. $b=\ln \left(B / X_{0}\right) / \sigma, u=\ln \left(U / X_{0}\right) / \sigma, \ell=\ln \left(L / X_{0}\right) / \sigma, h=u-\ell, u_{n}=u-2 n h$
$\ell_{n}=\ell-2 n h$
- Convenience variables: $v=\ln \left(V / X_{0}\right) / \sigma, \bar{v}=v / \sqrt{T}, \bar{v}_{i}=v / \sqrt{T_{i}}$
- Reflected variables: single barrier, $\check{v}=2 b-v$, double barriers, $\check{v}=2 u-v$

The $n$ -dimensional normal cumulative function is defined as
$$
\mathcal{N}_{n}\left(a_{1}, a_{2}, \cdots, a_{n} ; \Sigma\right)=\int_{-\infty}^{a_{1}} \int_{-\infty}^{a_{2}} \cdots \int_{-\infty}^{a_{n}} d^{n} x \phi_{n}(x ; \Sigma)
$$
where $\phi_{n}(x ; \Sigma)=\frac{e^{-\frac{1}{2} x^{T} \Sigma^{-1} x}}{\sqrt{(2 \pi)^{n} \operatorname{det} \Sigma}}$ is the $n$ -dimensional normal distribution density. We also use the notation $[a, b]$ to denote a finite integration domain, for example,
$$
\mathcal{N}_{n}\left(\left[a_{1}, b_{1}\right], b_{2}, \cdots, b_{n} ; \Sigma\right)=\int_{a_{1}}^{b_{1}} \int_{-\infty}^{b_{2}} \cdots \int_{-\infty}^{b_{n}} d^{n} x \phi_{n}(x ; \Sigma)
$$
These functions are computed numerically with Genz's MVNPACK.

The integral $I_{n}(x)$ can be computed numerically using any quadrature
subroutines. It can also be expressed explicitly in normal cumulatve distribution functionv of a
complex variable. We have $\left(\bar{x}=x / \sqrt{T}, \bar{\omega}_{n}=\omega_{n} \sqrt{T}\right)$
$$
\begin{aligned}
I_{n}(x) &=\operatorname{Im}\left\{e^{i \omega_{n} x}\left[\mathcal{N}\left(\bar{x}+i \bar{\omega}_{n}\right)-\mathcal{N}\left((\bar{x}-\bar{h})+i \bar{\omega}_{n}\right)\right]\right\} \\
&=\sin \left(\bar{\omega}_{n} \bar{x}\right)\left[F\left(\bar{x}, \bar{\omega}_{n}\right)-F\left(\bar{x}-\bar{h}, \bar{\omega}_{n}\right)\right]
\end{aligned}
$$

$$
+\cos \left(\bar{\omega}_{n} \bar{x}\right)\left[G\left(\bar{x}, \bar{\omega}_{n}\right)-G\left(\bar{x}-\bar{h}, \bar{\omega}_{n}\right)\right]
$$
where $\mathcal{N}(x+i y)=F(x, y)+i G(x, y)$ with
$$
\begin{aligned}
F(x, y)=& \mathcal{N}(x)+\frac{\phi(x)}{\sqrt{\pi}}\left\{\frac{1-\cos x y}{2 x}\right.\\
&\left.+\sum_{n=1}^{\infty} \frac{e^{-n^{2} / 4}}{n^{2}+2 x^{2}}\left(2 x-2 x \cosh \frac{n y}{\sqrt{2}} \cos x y+n \sqrt{2} \sinh \frac{n y}{\sqrt{2}} \sin x y\right)\right\} \\
G(x, y)=& \frac{\phi(x)}{\sqrt{\pi}}\left\{\frac{\sin x y}{2 x}+\sum_{n=1}^{\infty} \frac{e^{-n^{2} / 4}}{n^{2}+2 x^{2}}\left(2 x \cosh \frac{n y}{\sqrt{2}} \sin x y+n \sqrt{2} \sinh \frac{n y}{\sqrt{2}} \cos x y\right)\right\}
\end{aligned}
$$


# 3. Options - Black Scholes Assumption

KOC = knock-out call, UOP = up-and-out put.

## 3.1. Touch Options

$$V_{OT} + V_{NT} = V_F = \left\{
\begin{array}{lcl}
e^{-r_d T}       &      (DC)\\
e^{-r_f T}X_0     &      (FC)\\
\end{array} \right.
$$

## 3.2. Digital and Barrier Digital Options

Barrier Digital Options:

$$V_{BD}^{KOC} + V_{BD}^{KIC} = V_D^C, V_{BD}^{KOP} + V_{BD}^{KIP} = V_D^P, V_{BD}^{KOC} + V_{BD}^{KOP} = V_{NT}, V_{BD}^{KIC} + V_{BD}^{KIP} = V_{OT} $$

where digital option values are ($\epsilon = 1, -1$ for call and put)

$$V_D = \left\{
\begin{array}{lcl}
e^{-r_d T}  \Phi(\epsilon d_2)     &      (DC)\\
e^{-r_f T}X_0 \Phi(\epsilon d_1)    &      (FC)\\
\end{array} \right.
$$

## 3.3. Barrier Options

Barrier Options values are related to that of barrier digits.

$$V_B^{KIC/KOC} = V_{BD}^{KIC/KOC, FC} - KV_{BD}^{KIC/KOC, DC}, V_B^{KIP/KOP} = KV_{BD}^{KIP/KOP, DC} - V_{BD}^{KIP/KOP, FC}$$

## 3.4. Domestic-Foreign Symmetry

By changing the numeraire from the domestic cash bond $e^r d^t$ to the foreign counterpart $X_t e^r f^t$, it implies the domestic-foreigh symmetry for barrier digital and barrier options

$$V_{BD}^{FC} = X_0 V_{BD}^{DC} (\rm up \leftrightarrow \rm down, \rm call \leftrightarrow \rm put, \rm r_d \leftrightarrow \rm r_f, K \rightarrow \frac{X_0^2}{K}, B \rightarrow \frac{X_0^2}{B})$$

$$V_{BD}^{DC} = \frac{1}{X_0} V_{BD}^{FC} (\rm up \leftrightarrow \rm down, \rm call \leftrightarrow \rm put, \rm r_d \leftrightarrow \rm r_f, K \rightarrow \frac{X_0^2}{K}, B \rightarrow \frac{X_0^2}{B})$$

$$V_B = \frac{K}{X_0} V_B (\rm up \leftrightarrow \rm down, \rm call \leftrightarrow \rm put, \rm r_d \leftrightarrow \rm r_f, K \rightarrow \frac{X_0^2}{K}, B \rightarrow \frac{X_0^2}{B})$$

For single barrier digital, the down-and-out put is related to up-and-out call:

$$V_{BD}^{DOP, FC} (X_0, K, B, r_d, r_f) = X_0 V_{BD}^{UOC, DC} (X_0, \frac{X_0^2}{K}, \frac{X_0^2}{B}, r_f, r_d)$$

$$V_{BD}^{DOP, DC} (X_0, K, B, r_d, r_f) = \frac{1}{X_0}(X_0, \frac{X_0^2}{K}, \frac{X_0^2}{B}, r_f, r_d)$$

Summary: The central quantity is $V_{DB, DC} ^{UOC}$ for single barrier, and $V_{DB, DC}^{KOC}$ for double barrier. 

## 3.5. Notation

$$b = \ln (B/X_0)/\sigma, \bar{b}_T = \ln(B/X_0)\sigma \sqrt{T}, \bar{b}_i = \ln(B/X_0)/\sigma \sqrt{T_i}$$

$$\mu = \frac{r_d - r_f}{\sigma} - \frac{\sigma}{2}, \mu' = \mu + \sigma = \frac{r_d - r_f}{\sigma} + \frac{\sigma}{2}$$

$$\beta = \left\{
\begin{array}{lcl}
(B/X_0)^{2\mu/\sigma}    &      (\rm Single ~ \rm Barrier)\\
e^{-r_f T}X_0 \Phi(\epsilon d_1)    &      (FC)\\
\end{array} \right.
, \gamma = (U/L)^{2\mu / \sigma}$$

where $B, U, L$ are barrier levels, and for single Barrier define sign $s$ to be $1$ for up and $-1$ for down barrier.

# 4. Barrier Digital Call Values

## 4.1. Full Monitoring

- Single barrier: The UOC option value is 
  $$ V_{BD, DC} ^{UOC} = e^{-r_d T} \theta (\bar{b} - \bar{k}) \left[ \Phi(d_{\bar{k}, \bar{\mu}}, d_{\bar{b}, \bar{\mu}}) -\beta \Phi(d_{\pi(\bar{k}), \bar{\mu}}, d_{\pi(\bar{b}), \bar{\mu}}) \right]$$

  where $\pi$ is a reflection, $\pi : \bar{x} \mapsto \bar{x} - 2\bar{b}, \Phi(a,b) = \displaystyle{\int_a^b dx ~ \phi(x) = \Phi(b) - \Phi(a)}$, $\theta(x)$ is step function, $d_{x,y} = x - y$.

- Double barrier: KOC option value is
  $$V_{BD, DC}^{KOC} = e^{-r_d T}\theta (\bar{b} - \bar{k}) \sum_{n = - \infty} ^ {\infty} \gamma^n \left[ \Phi(d_{\bar{k}_n \vee \bar{l}_n, \bar{\mu}}, d_{\bar{u}_n, \bar{\mu}}) - \beta \Phi(d_{\pi_u (\bar{k}_n \vee \bar{l}_n), \bar{\mu}}, d_{\pi_u(\bar{u}_n), \bar{\mu}})
  \right]$$

  where $\bar{h} = \bar{u} - \bar{l}, \bar{x}_n = \bar{x} - 2n\bar{h}$, for $\bar{x} = \bar{u}$, and $\pi_u: \bar{x} \mapsto \bar{x} - 2\bar{u}, \pi_l: \bar{x} \mapsto \bar{x} - 2\bar{l}$.

## 4.2. Partial-Start Monitoring

Monitoring window = $[0, T_1]$.

- Single barrier: The UOC option value is
  $$V_{BD, DC}^{UOC} = e^{-r_d T} \left[ \Phi_2 (d_{\bar{b}_1, \bar{\mu}_1}, -d_{\bar{k}, \bar{\mu}}; -\rho) - \beta \Phi_2 (d_{\pi(\bar{b}_1), \bar{\mu}_1}, -d_{\pi(\bar{k}), \bar{\mu}} ; -\rho \right]$$

  where $\Phi_2$ is 2-dimensional normal cumulative function, $\rho = \sqrt{T_1 / T}$

- Double barrier: KOC option value is

  $$V_{BD, DC}^{KOC} = e^{-r_d T} \sum_{n = -\infty} ^ {\infty} \gamma^n \left[ \Phi_2 ([d_{\bar{l}_{1,n}, \bar{\mu}_1}, d_{\bar{u}_{1,n}, \bar{\mu}_1}], -d_{\bar{k}_n, \bar{\mu}}; -\rho) - \beta \Phi_2 (d_{\pi_u (\bar{l}_1, n), \bar{\mu}_1}, d_{\pi_u(\bar{k}_n), \bar{\mu}} ; -\rho) \right]$$

## 4.3. Windows Monitoring

Monitoring window $[T_1, T_2]$.

- Single barrier: The type I UOC option value is

 $$V_{\rm BD, \rm DC}^{\rm UOC, \rm I} = e^{-r_d T} \left[ \Phi_3 (d_{\bar{b}_1, \mu_1}, d_{\bar{b}_2, \bar{\mu}_2}, -d_{\bar{k}, \bar{\mu}}; \Sigma_3) -\beta \Phi_3(-d_{\pi(\bar{b}_1, \bar{\mu}_1)}, d_{\pi(\bar{b}_2), \bar{\mu}_2}, -d_{\pi(\bar{k}), \bar{\mu}}; \Sigma_2) \right]$$

 where $\Sigma_i = J_i \Sigma J_i^T, i = 1, 2, 3$, $J$ is inversion matrix, $J_1 = \rm diag (-1,1,1,1)$. $\Sigma$ is correlation matrix, with entries $\rho_{12} = \sqrt{T_1 / T_2}, \rho_{13} = \sqrt{T_1/T}, \rho_{23} = \sqrt{T_2/T}$.

 For Type II Option, the UOC option value is 

$$
\begin{array}{l}
 V_{\rm BD, \rm DC}^{\rm UOC, \rm II} = e^{-r_d T} \left[ \Phi_3 (d_{\bar{b}_1, \mu_1}, d_{\bar{b}_2, \bar{\mu}_2}, -d_{\bar{k}, \bar{\mu}}; \Sigma_3) + \Phi_3(-d_{\bar{b}_1, \bar{\mu}_1}, d_{\bar{b}_2, \bar{\mu}_2}, -d_{\bar{k}, \bar{\mu}}; \Sigma) \right] \\
- \beta \left( \Phi_3 (-d_{\pi(\bar{b}_1), \bar{\mu}_1},-d_{\pi(\bar{b}_2), \bar{\mu}_2} , -d_{\pi(\bar{k}), \bar{\mu}}; \Sigma_1 \right)
\end{array}
$$

- Double barrier option value is

$$
\begin{array}{l}
V_{\mathrm{BD}, \mathrm{DC}}^{\mathrm{KOC}}=e^{-r_{d} T} \sum_{n=-\infty}^{\infty} \gamma^{n}\left[\Phi_{3}\left(\left[d_{\bar{\ell}_{1}, \bar{\mu}_{1}}, d_{\bar{u}_{1}, \bar{\mu}_{1}}\right],\left[d_{\bar{\ell}_{2, n}, \bar{\mu}_{2}}, d_{\bar{u}_{2, n}, \bar{\mu}_{2}}\right],-d_{\bar{k}_{n}, \bar{\mu}} ; \Sigma_{3}\right)\right. \\
\left.-\beta \Phi_{3}\left(\left[-d_{\pi_{\ell}\left(\bar{\ell}_{1}\right), \bar{\mu}_{1}},-d_{\pi_{\mathrm{u}}\left(\bar{u}_{1}\right), \bar{\mu}_{1}}\right],\left[d_{\pi_{\mathrm{u}}\left(\bar{\ell}_{2, n}\right), \bar{\mu}_{2}}, d_{\pi_{u}\left(\bar{u}_{2, n}\right), \bar{\mu}_{2}}\right],-d_{\pi_{u}}\left(\bar{k}_{n}\right), \bar{\mu} ; \Sigma_{2}\right)\right]
\end{array}
$$

where 

$$\Phi_{3}([\mathrm{a}, \mathrm{b}],[\mathrm{c}, \mathrm{d}], \mathrm{e} ; \Sigma)=\int_{\mathrm{a}}^{\mathrm{b}} \mathrm{dx} \int_{\mathrm{c}}^{\mathrm{d}} \mathrm{dy} \int_{-\infty}^{\mathrm{e}} \mathrm{dz} ~ \phi_{3}(\mathrm{x}, \mathrm{y}, \mathrm{z} ; \Sigma)$$

## 4.4. European Monitoring

- Single barrier

$$\mathrm{V}_{\mathrm{DC}}^{\mathrm{UOC}}=\mathrm{e}^{-\mathrm{r}_{\mathrm{d}} \mathrm{T}} \theta(\overline{\mathrm{b}}-\overline{\mathrm{k}}) \Phi\left(\mathrm{d}_{\overline{\mathrm{k}}, \bar{\mu}}, \mathrm{d}_{\mathrm{b}, \bar{\mu}}\right), \quad \mathrm{V}_{\mathrm{DC}}^{\mathrm{NT}}=\mathrm{e}^{-\mathrm{r}_{\mathrm{d}} \mathrm{T}} \Phi\left(\mathrm{sd}_{\mathrm{b}, \bar{\mu}}\right)$$

- Double barrier

$$\mathrm{V}_{\mathrm{DC}}^{\mathrm{KOC}}=\mathrm{e}^{-\mathrm{r}_{\mathrm{d}} \mathrm{T}} \theta(\overline{\mathrm{b}}-\overline{\mathrm{k}}) \Phi\left(\mathrm{d}_{\mathrm{k} \vee \bar{\ell}, \bar{\mu}}, \mathrm{d}_{\mathrm{b}, \bar{\mu}}\right), \quad \mathrm{V}_{\mathrm{DC}}^{\mathrm{NT}}=\mathrm{e}^{-\mathrm{r}_{\mathrm{d}} \mathrm{T}} \Phi\left(\mathrm{d}_{\bar{\ell}, \bar{\mu}}, \mathrm{d}_{\overline{\mathrm{u}}, \bar{\mu}}\right)$$

## 4.5. One-Touch Pay-at-Hit

No touch option values can be obtained from that of barrier digitals by taking the limit $K \rightarrow 0$. One-touch pay-at-maturity can then be obtained from the no-touch by using the touch option identity. One-touch pay-at-hit options needs knowledge of first-passage time distributions.

## 4.6. Single Barrier

The one-touch value with full or partial start monitoring (F/PS) window $[0, T]$ is

$$\mathrm{V}_{\mathrm{DC}, \mathrm{F} / \mathrm{PS}}^{\mathrm{OT}}=\beta_{-} \Phi\left(-\mathrm{s} ~ \mathrm{d}_{\mathrm{b}, \bar{\alpha}}\right)+\beta_{+} \Phi\left(\mathrm{s} ~ \mathrm{d}_{\pi(\mathrm{b}), \bar{\alpha}}\right)$$

where 

$$
\alpha=\sqrt{\mu^{2}+2 r_{d}}, \bar{\alpha}=\alpha \sqrt{T}, \beta_{\pm}=e^{(\mu \pm \alpha)} b=\left(B / X_{0}\right)^{(\mu \pm \alpha) / \sigma}
$$

For partial-end or window (PE/W) barrier with monitoring window $[T_1, T]$, the one-touch option values are

$$\begin{aligned}
V_{\mathrm{DC}, \mathrm{PE} / \mathrm{W}}^{\mathrm{OT}, \mathrm{I}}=& e^{-r_{d} T_{1}} \Phi\left(-s d_{\bar{b}_{1}, \bar{\mu}_{1}}\right)+\beta_{-} \Phi_{2}\left(s d_{\bar{b}_{1}, \bar{\alpha}_{1}},-s d_{\bar{b}, \bar{\alpha}} ;-\rho\right) \\
&+\beta_{+} \Phi_{2}\left(-s d_{\pi\left(\bar{b}_{1}\right), \bar{\alpha}_{1}}, s d_{\pi(\bar{b}), \bar{\alpha}} ;-\rho\right)
\end{aligned}$$

$$\begin{array}{l}
\mathrm{V}_{\mathrm{DC}, \mathrm{PE} / \mathrm{W}}^{\mathrm{OT}, \mathrm{II}}=\beta_{-}\left[\Phi_{2}\left(\mathrm{~d}_{\mathrm{b}_{1}, \bar{\alpha}_{1}},-\mathrm{d}_{\mathrm{b}, \bar{\alpha}} ;-\rho\right)+\Phi_{2}\left(-\mathrm{d}_{\mathrm{b}_{1}, \bar{\alpha}_{1}}, \mathrm{~d}_{\mathrm{b}, \bar{\alpha}} ;-\rho\right)\right] \\
+\beta_{+}\left[\Phi_{2}\left(-\mathrm{d}_{\pi\left(\mathrm{b}_{1}\right), \bar{\alpha}_{1}}, \mathrm{~d}_{\pi(\mathrm{b}), \bar{\alpha}} ;-\rho\right)+\Phi_{2}\left(\mathrm{~d}_{\pi\left(\mathrm{b}_{1}\right), \bar{\alpha}_{1}},-\mathrm{d}_{\pi(\mathrm{E}), \bar{\alpha}} ;-\rho\right)\right]
\end{array}$$

Pay-foreign-cash-at-hit the one-touch values are

$$\mathrm{V}_{\mathrm{FC}, \mathrm{F} / \mathrm{PS}}^{\mathrm{OT}}=\mathrm{B} \mathrm{V}_{\mathrm{DC}, \mathrm{F} / \mathrm{PS}}^{\mathrm{OT}}, \quad \mathrm{V}_{\mathrm{FC}, \mathrm{PE} / \mathrm{W}}^{\mathrm{OT}}=\mathrm{BV}_{\mathrm{DC}, \mathrm{PE} / \mathrm{W}}^{\mathrm{OT}}$$

## 4.7. Double Barrier

The one-touch option value is

$$\mathrm{V}_{\mathrm{DC}, \mathrm{F} / \mathrm{PS}}^{\mathrm{OT}}=\mathrm{p} \mathrm{e}^{(\mu-\alpha) \mathrm{u}}+\mathrm{q} \mathrm{e}^{(\mu-\alpha) \ell}-\sum_{\mathrm{n}=1}^{\infty} \frac{2 \lambda_{\mathrm{n}} \omega_{\mathrm{n}}}{\left(\omega_{\mathrm{n}}^{2}+\alpha^{2}\right) \mathrm{h}} \mathrm{e}^{-\left(\omega_{\mathrm{n}}^{2}+\alpha^{2}\right) \mathrm{T} / 2} \sin \left(\omega_{\mathrm{n}} \mathrm{u}\right)$$

where 
$$\omega_{\mathrm{n}}=\mathrm{n} \pi / \mathrm{h}, \lambda_{\mathrm{n}}=\mathrm{e}^{\mu ~u}-(-)^{\mathrm{n}} \mathrm{e}^{\mu \ell} \text { and } \mathrm{p}=\left(\mathrm{e}^{2 \alpha h}-e^{2 \alpha u}\right) /\left(e^{2 \alpha h}-1\right), q=\left(e^{2 \alpha u}-1\right) /\left(e^{2 \alpha h}-1\right)$$

The first two terms on RHS are separated out to ensure fast convergence of the remaining sum. The result in limit $T \rightarrow \infty$ follows __optional sampling theorem__.

For partial-end or window barrier, the one-touch option value is 

$$\begin{aligned}
V_{D C, P E / W}^{O T}=e^{-r_{d} T_{1}}\left(\Phi\left(d_{\bar{\ell}_{1}, \bar{\mu}_{1}}\right)\right.&\left.+\Phi\left(-d_{\bar{u}_{1}, \bar{\mu}_{1}}\right)\right)+p_{\Phi} e^{(\mu-\alpha) u}+q_{\Phi} e^{(\mu-\alpha) \ell} \\
&-e^{-\alpha^{2} T / 2} \sum_{n=1}^{\infty} \frac{2 \lambda_{n} \omega_{n}}{\omega_{n}^{2}+\alpha^{2}} e^{-\omega_{n}^{2}\left(T-T_{1}\right) / 2} I_{n}(u / h)
\end{aligned}$$

where 
$$I_{n}(z)=\frac{1}{\sqrt{2 \pi} T_{1}} \int_{0}^{1} d x e^{-\frac{1}{2} \bar{h}_{1}^{2}(x-z)^{2}} \sin (n \pi x)$$
$$\begin{array}{c}
\mathrm{p}_{\Phi}=\left[\mathrm{e}^{2 \alpha \mathrm{h}} \Phi\left(\mathrm{d}_{\bar{\ell}_{1},-\bar{\alpha}_{1}}, \mathrm{~d}_{\bar{u}_{1},-\bar{\alpha}_{1}}\right)-\mathrm{e}^{2 \alpha \mathrm{u}} \Phi\left(\mathrm{d}_{\bar{\ell}_{1}, \bar{\alpha}_{1}}, \mathrm{~d}_{\overline{\mathrm{u}}_{1}, \bar{\alpha}_{1}}\right)\right] /\left(\mathrm{e}^{2 \alpha \mathrm{h}}-1\right) \\
\mathrm{q}_{\Phi}=\left[\mathrm{e}^{2 \alpha \mathrm{u}} \Phi\left(\mathrm{d}_{\bar{\ell}_{1}, \bar{\alpha}_{1}}, \mathrm{~d}_{\overline{\mathrm{u}}_{1}, \bar{\alpha}_{1}}\right)-\Phi\left(\mathrm{d}_{\bar{\ell}_{1},-\bar{\alpha}_{1}}, \mathrm{~d}_{\bar{u}_{1},-\bar{\alpha}_{1}}\right)\right] /\left(\mathrm{e}^{2 \alpha \mathrm{h}}-1\right)
\end{array}$$

---

For double barier pay-foreign-cash-at-hit one-touch options are

$$\mathrm{V}_{\mathrm{FC}, \mathrm{F} / \mathrm{PS}}^{\mathrm{OT}}=\mathrm{p} \mathrm{U} \mathrm{e}^{(\mu-\alpha) \mathrm{u}}+\mathrm{q} \mathrm{L} \mathrm{e}^{(\mu-\alpha) \ell}-\sum_{\mathrm{n}=1}^{\infty} \frac{2 \lambda_{\mathrm{n}}^{\prime} \omega_{\mathrm{n}}}{\left(\omega_{\mathrm{n}}^{2}+\alpha^{2}\right) \mathrm{h}} \mathrm{e}^{-\left(\omega_{\mathrm{n}}^{2}+\alpha^{2}\right) \mathrm{T} / 2} \sin \left(\omega_{\mathrm{n}} \mathrm{u}\right)$$

$$\begin{aligned}
V_{F C, P E / W}^{O T}=& e^{-r_{f} T_{1}} X_{0}\left(\Phi\left(d_{\bar{\ell}_{1}, \bar{\mu}_{1}^{\prime}}\right)+\Phi\left(-d_{\bar{u}_{1}, \bar{\mu}_{1}^{\prime}}\right)\right)+p_{\Phi} U_{e}^{(\mu-\alpha) u}+q_{\Phi} L e^{(\mu-\alpha) \ell} \\
&+e^{-\alpha^{2} T / 2} \sum_{n=1}^{\infty}(-)^{n} \frac{2 \lambda_{n}^{\prime} \omega_{n}}{\omega_{n}^{2}+\alpha^{2}} e^{-\omega_{n}^{2}\left(T-T_{1}\right) / 2} I_{n}(-\ell / h)
\end{aligned}$$

where 
$$\lambda_{\mathrm{n}}^{\prime}=\mathrm{e}^{\mu \mathrm{u}} \mathrm{U}-(-)^{\mathrm{n}} \mathrm{e}^{\mu \ell} \mathrm{L}$$

# 5. Vanna Volga Method for FX Exotics

Although not self-consistent and arbitrage-free, the Vanna-Volga method is popular among practitioners to price the first-generation exotics. 

The Vanna Volga attempts to hedge against implied volatility movements through vega $\nu = \frac{\partial V}{\partial \sigma}$, vanna $\nu_\beta = \frac{\partial^2 V}{\partial \sigma \partial S}$, volga $\nu_\gamma = \frac{\partial^2 V}{\partial \sigma^2}$, i.e. overhedges, by

$$X_{VV} = X_{BS} + \Omega^T Y_{BS}$$

where 

$$\Omega = A^{-1}C$$

$$A=\left(\begin{array}{lll}
\mathcal{V}^{\mathrm{ATM}} & \mathcal{V}_{\beta}^{\mathrm{ATM}} & \mathcal{V}_{\gamma}^{\mathrm{ATM}} \\
\mathcal{V}^{\mathrm{RR}} & \mathcal{V}_{\beta}^{\mathrm{RR}} & \mathcal{V}_{\gamma}^{\mathrm{RR}} \\
\mathcal{V}^{\mathrm{BF}} & \mathcal{V}_{\beta}^{\mathrm{BF}} & \mathcal{V}_{\gamma}^{\mathrm{BF}} 
\end{array}\right)_{BS}, \quad \mathrm{C}=\left(\begin{array}{c}
\mathrm{ATM} \operatorname{cost} \\
\mathrm{RR} \operatorname{cost} \\
\mathrm{BF} \operatorname{cost}
\end{array}\right)=\left(\begin{array}{c}
0 \\
V_{\mathrm{MKT}}^{\mathrm{RR}}-V_{\mathrm{BS}}^{\mathrm{RR}} \\
V_{\mathrm{MKT}}^{\mathrm{BF}}-V_{\mathrm{BS}}^{\mathrm{BF}}
\end{array}\right)$$

- $RR$ is $25 \Delta$ risk-reversal
- $BF$ is $25 \Delta$ butterfly
- Quantities with BS subscripts refer to those computed in Black-Scholes framework with single ATM volatility $\sigma_{ATM}$
- MKT refer to those computed with volatility skew assumptions.
- $Y_{BS}$ is Black-Scholes vega, vanna and volga for exotics, modified by fudge factors.
- $Y_{BS} = (p \nu, p_{\beta} \nu_\beta, p_\gamma \nu_\gamma)^T$.

Fudge factors for KO options are taken as barrier survivial probabilities:

- Single barrier:

$$\begin{array}{l}
\mathrm{F} / \mathrm{PS}: \Phi\left(s d_{\bar{b}, \bar{\mu}}\right)-\beta \Phi\left(s d_{\pi(\bar{b}), \bar{\mu}}\right) \\
\mathrm{PE} / \mathrm{W}: \Phi_{2}\left(s d_{\bar{b}_{1}, \bar{\mu}_{1}}, s d_{\bar{b}, \bar{\mu}} ; \rho\right)-\beta \Phi_{2}\left(-s d_{\pi\left(\bar{b}_{1}\right), \bar{\mu}_{1}}, s d_{\pi(\bar{b}), \bar{\mu}} ;-\rho\right)
\end{array}$$

- Double barrier

$$
\mathrm{F} / \mathrm{PS}: \sum_{n=-\infty}^{\infty} \gamma^{n}\left[\Phi\left(d_{\bar{\ell}_{n}, \bar{\mu}}, d_{\bar{u}_{n}, \bar{\mu}}\right)-\beta \Phi\left(d_{\pi_{u}\left(\bar{\ell}_{n}\right), \bar{\mu}}, d_{\pi_{u}\left(\bar{u}_{n}\right)}, \bar{\mu}\right)\right]
$$

$$
\begin{aligned}
\sum_{\mathrm{PE} / \mathrm{W}:}^{\infty} \gamma^{n} &\left[\Phi_{2}\left(\left[d_{\bar{\ell}_{1}, \bar{\mu}_{1}}, d_{\bar{u}_{1}, \bar{\mu}_{1}}\right],\left[d_{\bar{\ell}_{n}, \bar{\mu}}, d_{\bar{u}_{n}, \bar{\mu}}\right] ; \rho\right)\right.\\
&\left.-\beta \Phi_{2}\left(\left[-d_{\pi_{\ell}\left(\bar{\ell}_{1}\right), \bar{\mu}_{1}},-d_{\pi_{u}\left(\bar{u}_{1}\right)}, \bar{\mu}_{1}\right],\left[d_{\pi_{\mathrm{u}}\left(\bar{\ell}_{n}\right), \bar{\mu}}, d_{\pi_{u}\left(\bar{u}_{n}\right)}, \bar{\mu}\right] ;-\rho\right)\right]
\end{aligned}
$$

For plain vanilla option, the Vanna Volga price for an option of strike $K$ is the combination of 3 options:

$$
\mathrm{C}_{\mathrm{VV}}(\mathrm{K})=\mathrm{C}_{\mathrm{BS}}(\mathrm{K})+\sum_{\mathrm{i}=1}^{3} \mathrm{w}_{\mathrm{i}}(\mathrm{K})\left(\mathrm{C}_{\mathrm{MKT}}\left(\mathrm{K}_{\mathrm{i}}\right)-\mathrm{C}_{\mathrm{BS}}\left(\mathrm{K}_{\mathrm{i}}\right)\right)
$$

where 
$$
w_{\mathrm{i}}(\mathrm{K})=\frac{\mathcal{V}(\mathrm{K})}{\mathcal{V}\left(\mathrm{K}_{\mathrm{i}}\right)} \prod_{j=1, \mathrm{j} \neq \mathrm{i}}^{3} \frac{\ln \mathrm{K}_{\mathrm{j}} / \mathrm{K}}{\ln \mathrm{K}_{\mathrm{j}} / \mathrm{K}_{\mathrm{i}}}
$$

Note that $w_i(K_j) = \delta_{ij}$, so Vanna Volga prices interpolate market prices. 

This equation can be inverted to get implied volatilities, so it can also be interpolated as an interpolation of implied volatilities.

# 6. Adjustments

## 6.1. Volatility Time Adjustments

In trading system, we use volatility time to calculate volatility related variables, and use the interest time for other variables. So the volatility time is not the same as interest time, to incorporate this into formula, we can adjust the rates $r_d$ and $r_f$ with volatility time and interest time.

$$
r=r_{d} \frac{\tau_{0, e}}{\tau_{0, e}^{\nu}}, \quad y=r_{f} \frac{\tau_{0, e}}{\tau_{0, e}^{\nu}}
$$

We use the adjusted rates $r, y$ and $T = \tau_{0, e}^\nu$ for calculations. This adjustment is internal implementation, no time is actually adjusted.

## 6.2. Settle Adjustments

When trade settlement days and exercise settlement days of FX exotics are non-zero, the formulas needs further adjustment.

Define adjusted spot FX rate $X$

$$
\mathrm{X}=\mathrm{SAF} \times \mathrm{X}_{0}
$$

where for physical settlement,

$$\rm SAF = \frac{e^{(-r_d(t_0, t_{S_u}) - r_f (t_0, t_{S_u})) \tau(t_0, t_{S_u})}}{e^{(-r_d(t_e, t_{e+S_x}) - r_f (t_e, t_{e+S_x})) \tau(t_e, t_{e+S_x})}}$$

for cash settlement,

$$\rm SAF = \frac{e^{(-r_d(t_0, t_{S_u}) - r_f (t_0, t_{S_u})) \tau(t_0, t_{S_u})}}{e^{(-r_d(t_e, t_{e+S_u}) - r_f (t_e, t_{e+S_u})) \tau(t_e, t_{e+S_u})}}$$

To incorporate settlement, we can first replace $X_0$ with $X$, and then adjust the fair $V$ with exercise settle discount factors. The adjusted fair is

$$
V^{\prime}=V \times e^{-r_{d}\left(t_{e}, t_{e+S_{x}}\right) \tau\left(t_{e}, t_{c+5_{x}}\right)}
$$

# 7. References

[1] A. Genz, http://www.math.wsu.edu/faculty/genz/software/fort77/mvnpack.f

[2] A. Castagna, F.Mercurio, *The Vanna-Volga method for implied volatilities*, Risk (2007, Jan) 106-111

[3] F. Bossens, G.Raye, N.S. Skantzos, and G.Deelstra, *Vanna Volga methods applied to FX derivatives: from theory to market practice*, arXiv:0904.1074(2009).

