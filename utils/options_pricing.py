"""
Options pricing utilities using Black-Scholes model and Greeks calculations.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BlackScholesCalculator:
    """Black-Scholes options pricing calculator with Greeks."""
    
    def __init__(self):
        pass
    
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes."""
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes."""
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def call_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        
        Returns:
        Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    def put_price(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Black-Scholes put option price.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        
        Returns:
        Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put_price, 0)
    
    def delta_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option delta (price sensitivity to underlying)."""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    def delta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option delta."""
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma (delta sensitivity to underlying)."""
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def theta_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option theta (time decay)."""
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2))
        return theta / 365  # Convert to daily theta
    
    def theta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option theta."""
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
        return theta / 365  # Convert to daily theta
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega (volatility sensitivity)."""
        if T <= 0:
            return 0.0
        
        d1 = self._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% vol change
    
    def rho_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option rho (interest rate sensitivity)."""
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma)
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Divide by 100 for 1% rate change
    
    def rho_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option rho."""
        if T <= 0:
            return 0.0
        
        d2 = self._d2(S, K, T, r, sigma)
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Divide by 100 for 1% rate change
    
    def implied_volatility(self, option_price: float, S: float, K: float, 
                          T: float, r: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Parameters:
        option_price: Market price of the option
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance
        
        Returns:
        Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                price = self.call_price(S, K, T, r, sigma)
            else:
                price = self.put_price(S, K, T, r, sigma)
            
            vega_val = self.vega(S, K, T, r, sigma) * 100  # Convert back from percentage
            
            if abs(vega_val) < 1e-10:  # Avoid division by zero
                break
            
            price_diff = price - option_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            # Newton-Raphson update
            sigma = sigma - price_diff / vega_val
            
            # Keep sigma positive and reasonable
            sigma = max(0.001, min(sigma, 5.0))
        
        return sigma
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict:
        """Calculate all Greeks for an option."""
        if option_type.lower() == 'call':
            price = self.call_price(S, K, T, r, sigma)
            delta = self.delta_call(S, K, T, r, sigma)
            theta = self.theta_call(S, K, T, r, sigma)
            rho = self.rho_call(S, K, T, r, sigma)
        else:
            price = self.put_price(S, K, T, r, sigma)
            delta = self.delta_put(S, K, T, r, sigma)
            theta = self.theta_put(S, K, T, r, sigma)
            rho = self.rho_put(S, K, T, r, sigma)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': self.gamma(S, K, T, r, sigma),
            'theta': theta,
            'vega': self.vega(S, K, T, r, sigma),
            'rho': rho,
            'underlying_price': S,
            'strike_price': K,
            'time_to_expiry': T,
            'risk_free_rate': r,
            'volatility': sigma,
            'option_type': option_type
        }


class OptionsPortfolio:
    """Options portfolio analysis and risk management."""
    
    def __init__(self):
        self.calculator = BlackScholesCalculator()
        self.positions = []
    
    def add_position(self, symbol: str, option_type: str, strike: float, 
                    expiry: datetime, quantity: int, premium_paid: float,
                    underlying_price: float, volatility: float, risk_free_rate: float = 0.05):
        """Add an options position to the portfolio."""
        
        # Calculate time to expiry
        time_to_expiry = (expiry - datetime.now()).days / 365.0
        
        # Calculate current Greeks
        greeks = self.calculator.calculate_all_greeks(
            underlying_price, strike, time_to_expiry, risk_free_rate, 
            volatility, option_type
        )
        
        position = {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'quantity': quantity,
            'premium_paid': premium_paid,
            'current_price': greeks['price'],
            'greeks': greeks,
            'pnl': (greeks['price'] - premium_paid) * quantity * 100,  # Assuming 100 shares per contract
        }
        
        self.positions.append(position)
        return position
    
    def get_portfolio_greeks(self) -> Dict:
        """Calculate portfolio-level Greeks."""
        total_delta = sum(pos['greeks']['delta'] * pos['quantity'] for pos in self.positions)
        total_gamma = sum(pos['greeks']['gamma'] * pos['quantity'] for pos in self.positions)
        total_theta = sum(pos['greeks']['theta'] * pos['quantity'] for pos in self.positions)
        total_vega = sum(pos['greeks']['vega'] * pos['quantity'] for pos in self.positions)
        total_rho = sum(pos['greeks']['rho'] * pos['quantity'] for pos in self.positions)
        total_pnl = sum(pos['pnl'] for pos in self.positions)
        
        return {
            'total_delta': total_delta,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'total_vega': total_vega,
            'total_rho': total_rho,
            'total_pnl': total_pnl,
            'position_count': len(self.positions)
        }
    
    def get_risk_metrics(self) -> Dict:
        """Calculate portfolio risk metrics."""
        portfolio_greeks = self.get_portfolio_greeks()
        
        # Calculate maximum loss scenarios
        max_loss_1_day = portfolio_greeks['total_theta']  # Time decay
        max_loss_vol_shock = abs(portfolio_greeks['total_vega'] * 5)  # 5% vol shock
        
        return {
            'max_daily_theta_loss': max_loss_1_day,
            'max_vol_shock_loss': max_loss_vol_shock,
            'delta_exposure': portfolio_greeks['total_delta'],
            'gamma_risk': portfolio_greeks['total_gamma'],
            'vega_risk': portfolio_greeks['total_vega'],
        }


def calculate_option_price(S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str = 'call') -> Dict:
    """Convenience function to calculate option price and Greeks."""
    calculator = BlackScholesCalculator()
    return calculator.calculate_all_greeks(S, K, T, r, sigma, option_type)


def calculate_implied_vol(option_price: float, S: float, K: float, 
                        T: float, r: float, option_type: str = 'call') -> float:
    """Convenience function to calculate implied volatility."""
    calculator = BlackScholesCalculator()
    return calculator.implied_volatility(option_price, S, K, T, r, option_type)


if __name__ == "__main__":
    # Test the module
    calculator = BlackScholesCalculator()
    
    # Test parameters
    S = 100  # Current stock price
    K = 105  # Strike price
    T = 0.25  # 3 months to expiry
    r = 0.05  # 5% risk-free rate
    sigma = 0.2  # 20% volatility
    
    print("Testing Black-Scholes Calculator...")
    
    # Calculate call option
    call_greeks = calculator.calculate_all_greeks(S, K, T, r, sigma, 'call')
    print(f"Call option price: ${call_greeks['price']:.2f}")
    print(f"Call delta: {call_greeks['delta']:.4f}")
    print(f"Gamma: {call_greeks['gamma']:.4f}")
    print(f"Theta: ${call_greeks['theta']:.2f}")
    print(f"Vega: ${call_greeks['vega']:.2f}")
    
    # Calculate put option
    put_greeks = calculator.calculate_all_greeks(S, K, T, r, sigma, 'put')
    print(f"\nPut option price: ${put_greeks['price']:.2f}")
    print(f"Put delta: {put_greeks['delta']:.4f}")
    
    # Test implied volatility
    market_price = 3.50
    implied_vol = calculator.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"\nImplied volatility for ${market_price} call: {implied_vol:.2%}")
    
    # Test portfolio
    portfolio = OptionsPortfolio()
    expiry_date = datetime.now() + timedelta(days=90)
    
    portfolio.add_position('AAPL', 'call', 105, expiry_date, 10, 3.0, 100, 0.2)
    portfolio.add_position('AAPL', 'put', 95, expiry_date, -5, 2.5, 100, 0.2)
    
    portfolio_greeks = portfolio.get_portfolio_greeks()
    print(f"\nPortfolio Greeks:")
    print(f"Total Delta: {portfolio_greeks['total_delta']:.2f}")
    print(f"Total PnL: ${portfolio_greeks['total_pnl']:.2f}")

