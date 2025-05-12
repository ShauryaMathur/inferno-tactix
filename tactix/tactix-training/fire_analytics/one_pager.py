# Simple Fire Threat Assessment Report Generator
# This script generates a professional PDF report based on fire simulation data
# Uses built-in fonts only, no external font files required

import json
import os
import datetime
import argparse
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

# Define GPS coordinate constants for your simulation area
# These define the real-world bounding box of your simulation grid
LAT_MIN = 34.05  # Southern boundary (latitude)
LAT_MAX = 34.15  # Northern boundary (latitude)
LON_MIN = -118.50  # Western boundary (longitude)
LON_MAX = -118.40  # Eastern boundary (longitude)

GRID_WIDTH = 240
GRID_HEIGHT = 160

class SimpleFireReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        
    def header(self):
        # Logo
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'FIRE THREAT ASSESSMENT REPORT', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def chapter_body(self, txt):
        self.set_font('Helvetica', '', 11)
        self.multi_cell(0, 5, txt)
        self.ln()
        
    def data_row(self, label, value, unit=""):
        self.set_font('Helvetica', 'B', 10)
        self.cell(80, 6, label, 0, 0)
        self.set_font('Helvetica', '', 10)
        self.cell(60, 6, f"{value} {unit}", 0, 1)

def grid_to_gps(grid_x, grid_y):
    """Convert grid coordinates to GPS coordinates"""
    # Linear interpolation from grid space to GPS space
    lon = LON_MIN + (grid_x / GRID_WIDTH) * (LON_MAX - LON_MIN)
    lat = LAT_MIN + (1 - grid_y / GRID_HEIGHT) * (LAT_MAX - LAT_MIN)  # Inverse Y-axis
    return lat, lon

def generate_burnt_area_chart(episodes_data, filename='burnt_area_chart.png'):
    """Generate a chart showing burnt area across episodes"""
    episode_nums = [i+1 for i in range(len(episodes_data))]
    burnt_areas = [ep["final_burnt_area"] for ep in episodes_data]
    
    plt.figure(figsize=(10, 5))
    plt.bar(episode_nums, burnt_areas, color='firebrick')
    plt.title('Burnt Area by Episode', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Burnt Area (cells)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    return filename

def generate_helitack_heatmap(episodes_data, filename='helitack_heatmap.png'):
    """Generate a heatmap of helitack operations"""
    all_points = []
    for episode in episodes_data:
        for coord in episode["helitack_coordinates"]:
            all_points.append((coord[1], coord[2]))  # x, y
    
    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        [p[0] for p in all_points],
        [p[1] for p in all_points],
        bins=[24, 16],  # Adjust bin size as needed
        range=[[0, 240], [0, 160]]
    )
    
    plt.figure(figsize=(10, 7))
    plt.imshow(heatmap.T, origin='lower', aspect='auto', 
               extent=[0, 240, 0, 160], cmap='hot')
    plt.colorbar(label='Number of Helitack Operations')
    plt.title('Helitack Operations Heatmap', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    return filename

def calculate_fire_statistics(episodes_data):
    """Calculate comprehensive fire statistics from the episode data"""
    total_episodes = len(episodes_data)
    total_burnt_area = sum(ep["final_burnt_area"] for ep in episodes_data)
    avg_burnt_area = total_burnt_area / total_episodes
    max_burnt_area = max(ep["final_burnt_area"] for ep in episodes_data)
    min_burnt_area = min(ep["final_burnt_area"] for ep in episodes_data)
    
    # Containment statistics
    containment_times = [ep["summary"]["containment_time"] for ep in episodes_data]
    avg_containment_time = sum(containment_times) / total_episodes
    max_containment_time = max(containment_times)
    min_containment_time = min(containment_times)
    
    # Helitack statistics
    helitack_counts = [ep["helitack_operations"] for ep in episodes_data]
    total_helitack_ops = sum(helitack_counts)
    avg_helitack_ops = total_helitack_ops / total_episodes
    
    # Calculate area per helitack operation
    efficiency = [ep["final_burnt_area"] / max(1, ep["helitack_operations"]) for ep in episodes_data]
    avg_efficiency = sum(efficiency) / total_episodes
    
    # Risk assessment based on statistics
    if avg_burnt_area > 30000:
        risk_level = "SEVERE"
        risk_color = (255, 0, 0)  # Red
    elif avg_burnt_area > 20000:
        risk_level = "HIGH"
        risk_color = (255, 128, 0)  # Orange
    elif avg_burnt_area > 10000:
        risk_level = "MODERATE"
        risk_color = (255, 255, 0)  # Yellow
    else:
        risk_level = "LOW"
        risk_color = (0, 255, 0)  # Green
    
    return {
        "total_episodes": total_episodes,
        "total_burnt_area": total_burnt_area,
        "avg_burnt_area": avg_burnt_area,
        "max_burnt_area": max_burnt_area,
        "min_burnt_area": min_burnt_area,
        "avg_containment_time": avg_containment_time,
        "max_containment_time": max_containment_time,
        "min_containment_time": min_containment_time,
        "total_helitack_ops": total_helitack_ops,
        "avg_helitack_ops": avg_helitack_ops,
        "avg_efficiency": avg_efficiency,
        "risk_level": risk_level,
        "risk_color": risk_color
    }

def find_most_critical_locations(episodes_data, top_n=3):
    """Find the most critical locations based on helitack operations"""
    # Aggregate all helitack coordinates
    all_coords = {}
    for episode in episodes_data:
        for coord in episode["helitack_coordinates"]:
            grid_x, grid_y = coord[1], coord[2]
            key = f"{grid_x},{grid_y}"
            all_coords[key] = all_coords.get(key, 0) + 1
    
    # Sort by frequency
    sorted_coords = sorted(all_coords.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N critical locations
    critical_locations = []
    for i in range(min(top_n, len(sorted_coords))):
        key, count = sorted_coords[i]
        grid_x, grid_y = map(int, key.split(','))
        lat, lon = grid_to_gps(grid_x, grid_y)
        critical_locations.append({
            "grid_x": grid_x,
            "grid_y": grid_y,
            "lat": lat,
            "lon": lon,
            "frequency": count
        })
    
    return critical_locations

def generate_report(data_file, output_pdf="Fire_Threat_Assessment_Report.pdf"):
    """Generate a comprehensive fire threat assessment report PDF"""
    
    # Load and parse the JSON data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    episodes_data = data["episodes"]
    
    # Calculate statistics
    stats = calculate_fire_statistics(episodes_data)
    critical_locations = find_most_critical_locations(episodes_data)
    
    # Generate charts
    burnt_area_chart = generate_burnt_area_chart(episodes_data)
    helitack_heatmap = generate_helitack_heatmap(episodes_data)
    
    # Create PDF report
    pdf = SimpleFireReportPDF()
    pdf.alias_nb_pages()
    
    # Title and Date
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, "", 0, 1)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%B %d, %Y')}", 0, 1)
    
    # Executive Summary
    pdf.chapter_title("EXECUTIVE SUMMARY")
    summary_text = (
        f"This report presents the analysis of {stats['total_episodes']} fire simulation episodes covering "
        f"a wildland area of approximately {GRID_WIDTH * GRID_HEIGHT / 100:.1f} square kilometers. "
        f"The overall fire risk assessment level is {stats['risk_level']}.\n\n"
        f"Across all simulations, an average of {stats['avg_burnt_area']:.1f} cells were burnt per episode, "
        f"with containment achieved after an average of {stats['avg_containment_time']:.1f} simulation steps. "
        f"Fire suppression operations deployed an average of {stats['avg_helitack_ops']:.1f} helitack drops per episode."
    )
    pdf.chapter_body(summary_text)
    
    # Key Metrics
    pdf.chapter_title("KEY METRICS")
    pdf.data_row("Total Simulation Episodes:", f"{stats['total_episodes']}")
    pdf.data_row("Average Burnt Area:", f"{stats['avg_burnt_area']:.1f}", "cells")
    pdf.data_row("Maximum Burnt Area:", f"{stats['max_burnt_area']}", "cells")
    pdf.data_row("Average Containment Time:", f"{stats['avg_containment_time']:.1f}", "steps")
    pdf.data_row("Total Helitack Operations:", f"{stats['total_helitack_ops']}")
    pdf.data_row("Average Operations per Episode:", f"{stats['avg_helitack_ops']:.1f}")
    pdf.data_row("Area per Helitack Operation:", f"{stats['avg_efficiency']:.1f}", "cells/operation")
    
    # Risk Assessment
    pdf.chapter_title("RISK ASSESSMENT")
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(*stats['risk_color'])
    pdf.cell(0, 10, f"Overall Risk Level: {stats['risk_level']}", 0, 1)
    pdf.set_text_color(0, 0, 0)  # Reset text color
    
    pdf.set_font('Helvetica', '', 11)
    risk_text = (
        f"The risk assessment is based on average burnt area, containment time, and operational efficiency. "
        f"The {stats['risk_level']} risk level indicates that this area "
    )
    
    if stats['risk_level'] == "SEVERE":
        risk_text += "requires immediate attention and enhanced fire prevention measures. Additional resources should be allocated for rapid response capabilities."
    elif stats['risk_level'] == "HIGH":
        risk_text += "requires careful monitoring and increased preparedness during fire seasons. Pre-positioning of resources is recommended."
    elif stats['risk_level'] == "MODERATE":
        risk_text += "should maintain standard fire prevention protocols with regular patrols during high-risk seasons."
    else:
        risk_text += "has effective fire management systems in place, but continued vigilance is advised."
    
    pdf.multi_cell(0, 5, risk_text)
    pdf.ln(5)
    
    # Critical Locations
    pdf.chapter_title("CRITICAL LOCATIONS")
    pdf.chapter_body("The following locations had the highest frequency of helitack operations, indicating hotspots that may require special attention for fire prevention and rapid response:")
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(20, 6, "Rank", 1)
    pdf.cell(30, 6, "Grid (X,Y)", 1)
    pdf.cell(60, 6, "GPS Coordinates", 1)
    pdf.cell(40, 6, "Frequency", 1)
    pdf.cell(40, 6, "Risk Level", 1)
    pdf.ln()
    
    pdf.set_font('Helvetica', '', 10)
    for i, loc in enumerate(critical_locations):
        # Determine risk level for this location
        if loc["frequency"] > 10:
            loc_risk = "High"
        elif loc["frequency"] > 5:
            loc_risk = "Moderate"
        else:
            loc_risk = "Low"
            
        pdf.cell(20, 6, f"{i+1}", 1)
        pdf.cell(30, 6, f"({loc['grid_x']},{loc['grid_y']})", 1)
        pdf.cell(60, 6, f"{loc['lat']:.6f}, {loc['lon']:.6f}", 1)
        pdf.cell(40, 6, f"{loc['frequency']} operations", 1)
        pdf.cell(40, 6, f"{loc_risk}", 1)
        pdf.ln()
    
    # Add charts
    pdf.add_page()
    pdf.chapter_title("FIRE ANALYSIS CHARTS")
    
    # Add burnt area chart
    pdf.image(burnt_area_chart, x=10, y=40, w=180)
    pdf.ln(70)  # Space for the image
    
    # Add helitack heatmap
    pdf.image(helitack_heatmap, x=10, y=120, w=180)
    pdf.ln(90)  # Space for the image
    
    # Recommendations
    pdf.add_page()
    pdf.chapter_title("RECOMMENDATIONS")
    
    recommendations = [
        f"Increase monitoring in the identified critical areas, especially at coordinates {critical_locations[0]['lat']:.6f}, {critical_locations[0]['lon']:.6f}.",
        f"Deploy additional resources during peak fire seasons based on the {stats['risk_level']} risk assessment.",
        "Implement preventative fire breaks in the most frequently affected regions.",
        f"Optimize helitack response strategies based on the typical {stats['avg_containment_time']:.1f} step containment time.",
        "Conduct regular drills to ensure rapid deployment to the identified critical areas.",
        "Review and update fire prevention protocols based on the simulation findings."
    ]
    
    for i, rec in enumerate(recommendations):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 6, f"Recommendation {i+1}:", 0, 1)
        pdf.set_font('Helvetica', '', 11)
        pdf.multi_cell(0, 5, rec)
        pdf.ln(3)
    
    # Conclusion
    pdf.chapter_title("CONCLUSION")
    conclusion_text = (
        f"The fire simulation analysis revealed a {stats['risk_level']} level of wildfire risk in the studied area. "
        f"With an average burnt area of {stats['avg_burnt_area']:.1f} cells across {stats['total_episodes']} episodes, "
        f"this region requires appropriate fire management strategies. The analysis identified {len(critical_locations)} "
        f"critical locations that serve as priority areas for preventative measures and rapid response planning.\n\n"
        f"By implementing the recommended actions, fire management authorities can enhance preparedness and reduce "
        f"potential damage from wildfires in this region."
    )
    pdf.chapter_body(conclusion_text)
    
    # Save the PDF
    pdf.output(output_pdf)
    
    # Clean up temporary files
    os.remove(burnt_area_chart)
    os.remove(helitack_heatmap)
    
    print(f"Report generated successfully: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a Fire Threat Assessment Report')
    # parser.add_argument('data_file', help='Path to the fire simulation JSON data file')
    parser.add_argument('--output', '-o', default='Fire_Threat_Assessment_Report.pdf', 
                        help='Output PDF filename')
    
    args = parser.parse_args()
    generate_report('./combined_fire_assessment.json', args.output)