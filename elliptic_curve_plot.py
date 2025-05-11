import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class EllipticCurve:
    """
    A class representing an elliptic curve of the form y^2 = x^3 + ax + b over a finite field F_p.
    """
    def __init__(self, a: int, b: int, p: int):
        """
        Initialize an elliptic curve with parameters a, b, and modulus p.
        
        Args:
            a: coefficient of x
            b: constant term
            p: prime modulus defining the field F_p
        """
        # Check that 4a^3 + 27b^2 != 0 (mod p) to ensure the curve is non-singular
        discriminant = (4 * (a**3) + 27 * (b**2)) % p
        if discriminant == 0:
            raise ValueError(f"The curve y^2 = x^3 + {a}x + {b} (mod {p}) is singular")
            
        self.a = a
        self.b = b
        self.p = p
        
    def is_on_curve(self, point: Tuple[int, int]) -> bool:
        """Check if a point (x, y) lies on the curve."""
        if point is None:  # Check if point is the point at infinity
            return True
            
        x, y = point
        left = (y * y) % self.p
        right = (x**3 + self.a * x + self.b) % self.p
        return left == right
        
    def add_points(self, P: Optional[Tuple[int, int]], Q: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Add two points on the elliptic curve.
        
        Args:
            P: First point as (x, y) or None for point at infinity
            Q: Second point as (x, y) or None for point at infinity
            
        Returns:
            The sum of P and Q, or None for the point at infinity
        """
        # Handle point at infinity cases
        if P is None:
            return Q
        if Q is None:
            return P
            
        x1, y1 = P
        x2, y2 = Q
        
        # Ensure points are on the curve
        if not self.is_on_curve(P) or not self.is_on_curve(Q):
            raise ValueError("Points must be on the curve")
            
        # If Q is the additive inverse of P
        if x1 == x2 and (y1 + y2) % self.p == 0:
            return None  # Point at infinity
            
        # Calculate the slope
        if x1 == x2 and y1 == y2:  # Point doubling
            # s = (3x^2 + a) / (2y) mod p
            numerator = (3 * (x1**2) + self.a) % self.p
            denominator = (2 * y1) % self.p
            # Find modular inverse of denominator
            inv_denominator = pow(denominator, self.p - 2, self.p)  # Fermat's little theorem
            s = (numerator * inv_denominator) % self.p
        else:  # Point addition
            # s = (y2 - y1) / (x2 - x1) mod p
            numerator = (y2 - y1) % self.p
            denominator = (x2 - x1) % self.p
            # Find modular inverse of denominator
            inv_denominator = pow(denominator, self.p - 2, self.p)  # Fermat's little theorem
            s = (numerator * inv_denominator) % self.p
            
        # Calculate the new point
        x3 = (s**2 - x1 - x2) % self.p
        y3 = (s * (x1 - x3) - y1) % self.p
        
        return (x3, y3)
    
    def scalar_multiply(self, k: int, P: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Compute the scalar multiplication k*P using the double-and-add algorithm.
        
        Args:
            k: A non-negative integer
            P: A point on the curve
            
        Returns:
            The result of k*P, or None for the point at infinity
        """
        if k == 0 or P is None:
            return None  # Point at infinity
            
        if k < 0:
            # For negative k, calculate -k*P as k*(-P)
            # Since -P = (x, -y mod p), we invert y
            x, y = P
            P = (x, (-y) % self.p)
            k = -k
            
        result = None  # Start with the point at infinity
        addend = P
        
        while k:
            if k & 1:  # If the least significant bit is 1
                result = self.add_points(result, addend)
            addend = self.add_points(addend, addend)  # Double the point
            k >>= 1  # Shift k right by 1
            
        return result
    
    def find_points(self) -> List[Tuple[int, int]]:
        """
        Find all points on the elliptic curve in F_p.
        
        Returns:
            A list of (x, y) points on the curve
        """
        points = []
        
        # Check each possible x-coordinate
        for x in range(self.p):
            # Calculate the right side of the equation: x^3 + ax + b
            rhs = (x**3 + self.a * x + self.b) % self.p
            
            # Try to find y such that y^2 = rhs (mod p)
            # For prime p, we can check if rhs is a quadratic residue
            # If y is a solution, then p-y is also a solution
            
            # Simple approach: check all possible y values
            for y in range(self.p):
                if (y * y) % self.p == rhs:
                    points.append((x, y))
        
        return points

    def plot_curve(self, title: str = None):
        """
        Create a scatter plot of all points on the elliptic curve.
        
        Args:
            title: Optional title for the plot
        """
        points = self.find_points()
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # Create plot
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, color='blue', alpha=0.7)
        
        # Plot settings
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Set plot title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f"Points on Elliptic Curve $y^2 = x^3 + {self.a}x + {self.b}$ over $\mathbb{{F}}_{{{self.p}}}$")
        
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Set equal aspect ratio
        plt.axis('equal')
        
        # Show the plot
        plt.show()


if __name__ == "__main__":
    # Example 1: Small curve with p=23, a=1, b=1
    curve1 = EllipticCurve(1, 1, 23)
    print(f"Number of points on curve 1: {len(curve1.find_points())}")
    curve1.plot_curve()
    
    # Example 2: Another small curve with p=17, a=2, b=2
    curve2 = EllipticCurve(2, 2, 17)
    print(f"Number of points on curve 2: {len(curve2.find_points())}")
    curve2.plot_curve() 