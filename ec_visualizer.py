#!/usr/bin/env python3

import argparse
from elliptic_curve_plot import EllipticCurve
from elliptic_curve_visualize import (
    visualize_subgroups,
    visualize_scalar_multiplication,
    visualize_pollard_rho
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Elliptic Curve Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Original default curve: y^2 = x^3 + 2x + 2 (mod 17)
    # New default curve: y^2 = x^3 + 3x + 7 (mod 29)
    parser.add_argument("--a", type=int, default=3, help="Coefficient a in y^2 = x^3 + ax + b")
    parser.add_argument("--b", type=int, default=7, help="Coefficient b in y^2 = x^3 + ax + b")
    parser.add_argument("--p", type=int, default=29, help="Prime modulus p defining the field F_p")
    
    parser.add_argument("--point", type=str, 
                      help="Base point for scalar multiplication (format: 'x,y')")
    parser.add_argument("--target", type=str, 
                      help="Target point for Pollard's Rho (format: 'x,y')")
    parser.add_argument("--k", type=int, default=5, 
                      help="Scalar value for multiplication (if target not provided)")
    
    parser.add_argument("--max-scalar", type=int, default=10, 
                      help="Maximum scalar value for scalar multiplication visualization")
    parser.add_argument("--max-iterations", type=int, default=20, 
                      help="Maximum iterations for Pollard's Rho visualization")
    
    visualization_group = parser.add_argument_group("Visualizations")
    visualization_group.add_argument("--all", action="store_true",
                                   help="Run all visualizations")
    visualization_group.add_argument("--plot", action="store_true",
                                   help="Basic scatter plot of curve points")
    visualization_group.add_argument("--subgroups", action="store_true",
                                   help="Visualize subgroups of the curve")
    visualization_group.add_argument("--scalar-mult", action="store_true",
                                   help="Visualize scalar multiplication")
    visualization_group.add_argument("--pollard-rho", action="store_true",
                                   help="Visualize Pollard's Rho algorithm")
    
    return parser.parse_args()


def parse_point(point_str):
    """Parse a point from 'x,y' format to tuple (x, y)"""
    if not point_str:
        return None
    try:
        x, y = map(int, point_str.split(','))
        return (x, y)
    except ValueError:
        raise ValueError(f"Invalid point format: {point_str}. Use 'x,y' format.")


def find_generator(curve):
    """Find a point of high order on the curve"""
    points = curve.find_points()
    if not points:
        return None
        
    # Start with first point
    best_point = points[0]
    best_order = 1
    
    for point in points:
        # Compute the order of the point
        current = point
        order = 1
        while True:
            current = curve.add_points(current, point)
            if current is None or current == point:
                break
            order += 1
            
        if order > best_order:
            best_point = point
            best_order = order
    
    return best_point, best_order


def main():
    args = parse_args()
    
    # Check if any visualization flags are set explicitly
    vis_flags = [args.plot, args.subgroups, args.scalar_mult, args.pollard_rho]
    if not args.all and not any(vis_flags):
        # If no flags set, enable all visualizations
        args.all = True
        
    print("🔶 Fancy Elliptic Curve Visualization 🔶")
    print("----------------------------------------")
    
    # Create the elliptic curve
    try:
        # Original curve (commented out)
        # curve = EllipticCurve(args.a, args.b, args.p)  # Default: y^2 = x^3 + 2x + 2 (mod 17)
        
        # New, fancier curve
        curve = EllipticCurve(args.a, args.b, args.p)  # Default: y^2 = x^3 + 3x + 7 (mod 29)
        print(f"Using curve: y^2 = x^3 + {args.a}x + {args.b} (mod {args.p})")
    except ValueError as e:
        print(f"Error creating elliptic curve: {e}")
        return
    
    # Find points on the curve
    points = curve.find_points()
    print(f"Found {len(points)} points on the curve")
    
    # Parse base point or find a generator
    base_point = parse_point(args.point)
    if base_point is None and (args.all or args.scalar_mult or args.pollard_rho):
        base_point, order = find_generator(curve)
        if base_point:
            print(f"Using point {base_point} of order {order} as base point")
        else:
            print("No points found on the curve. Exiting.")
            return
    
    # Check if base point is on the curve
    if base_point is not None and not curve.is_on_curve(base_point):
        print(f"Error: Point {base_point} is not on the curve")
        return
    
    # Parse target point or compute from base point
    target_point = parse_point(args.target)
    if target_point is None and (args.all or args.pollard_rho):
        if base_point:
            target_point = curve.scalar_multiply(args.k, base_point)
            print(f"Using target point {target_point} (= {args.k} × {base_point})")
    
    # Check if target point is on the curve
    if target_point is not None and not curve.is_on_curve(target_point):
        print(f"Error: Point {target_point} is not on the curve")
        return
    
    # Run visualizations based on flags
    if args.all or args.plot:
        print("\n📊 Plotting all points on the curve...")
        curve.plot_curve()
    
    if args.all or args.subgroups:
        print("\n🔄 Visualizing subgroups...")
        visualize_subgroups(curve)
    
    if (args.all or args.scalar_mult) and base_point:
        print(f"\n✖️ Visualizing scalar multiplication with base point {base_point}...")
        visualize_scalar_multiplication(curve, base_point, args.max_scalar)
    
    if (args.all or args.pollard_rho) and base_point and target_point:
        print(f"\n🔄 Visualizing Pollard's Rho with generator {base_point} and target {target_point}...")
        visualize_pollard_rho(curve, base_point, target_point, args.max_iterations)


if __name__ == "__main__":
    main() 