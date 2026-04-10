import os
import pandas as pd
import numpy as np
from pathlib import Path

def create_dataset():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    # Load national metrics
    pop_df = pd.read_excel(raw_dir / "Sri_Lanka_Population_1950_2025.xlsx")
    birth_df = pd.read_excel(raw_dir / "Sri_Lanka_Birth_Rate_1950_2025.xlsx")
    
    # Read death rate which is a csv
    death_df = pd.read_csv(raw_dir / "Year,Deaths per 1000 People.csv")
    
    density_df = pd.read_csv(raw_dir / "Sri-Lanka-Population-Density-People-per-Square-KM-2026-03-05-21-40.csv")
    rural_df = pd.read_excel(raw_dir / "Sri_Lanka_Rural_Population_1960_2023.xlsx")
    urban_df = pd.read_excel(raw_dir / "Sri_Lanka_Urban_Population_1960_2023.xlsx")

    district_df = pd.read_csv(raw_dir / "sri_lanka_district_population_2014_2024_new.csv")
    
    # 1. Prepare National Data
    print("Merging National data...")
    # pop_df columns: Year, Population
    # birth_df columns: Year, Birth Rate (per 1000 people)
    # death_df columns: Year, Deaths per 1000 People
    # density_df columns: Year, Population Density
    # rural_df columns: Year, Rural Population (might vary, let's just assume `Rural Population`)
    # urban_df columns: Year, Urban Population
    
    # To be safe, standardizing names:
    pop_df.columns = ["Year", "National_Population"]
    birth_df.columns = ["Year", "Birth_Rate"]
    death_df.columns = ["Year", "Death_Rate"]
    density_df.columns = ["Year", "Population_Density"]
    rural_df.columns = ["Year", "Rural_Population"]
    urban_df.columns = ["Year", "Urban_Population"]

    nat_df = pop_df.merge(birth_df, on="Year", how="outer")
    nat_df = nat_df.merge(death_df, on="Year", how="outer")
    nat_df = nat_df.merge(density_df, on="Year", how="outer")
    nat_df = nat_df.merge(rural_df, on="Year", how="outer")
    nat_df = nat_df.merge(urban_df, on="Year", how="outer")
    
    nat_df = nat_df.sort_values("Year").reset_index(drop=True)
    
    # Interpolate for missing data (like rural/urban which only go to 2023)
    # Use polynomial/linear to fill simple gaps
    nat_df = nat_df.interpolate(method='linear', limit_direction='both')

    # Convert yearly data to monthly data to support fine-grained predictions
    print("Interpolating yearly data into monthly...")
    months = np.arange(1, 13)
    monthly_data = []

    for i in range(len(nat_df) - 1):
        year1_row = nat_df.iloc[i]
        year2_row = nat_df.iloc[i+1]
        
        y = year1_row["Year"]
        
        for m in months:
            # Fraction of the year passed
            frac = (m - 1) / 12.0
            
            # Linear interpolation between year and year+1
            interp_row = {"Year": y, "Month": m}
            interp_row["Date"] = pd.to_datetime(f"{int(y)}-{int(m):02d}-01")
            
            for col in nat_df.columns:
                if col != "Year":
                    val = year1_row[col] + frac * (year2_row[col] - year1_row[col])
                    interp_row[col] = val
                    
            monthly_data.append(interp_row)

    # For the last year, we can just extrapolate forward using the previous year's difference
    last_year_row = nat_df.iloc[-1]
    prev_year_row = nat_df.iloc[-2]
    y = last_year_row["Year"]
    for m in months:
        frac = (m - 1) / 12.0
        interp_row = {"Year": y, "Month": m}
        interp_row["Date"] = pd.to_datetime(f"{int(y)}-{int(m):02d}-01")
        for col in nat_df.columns:
            if col != "Year":
                # Linear extrapolation
                diff = last_year_row[col] - prev_year_row[col]
                interp_row[col] = last_year_row[col] + frac * diff
        monthly_data.append(interp_row)

    monthly_nat_df = pd.DataFrame(monthly_data)
    monthly_nat_df.to_csv(processed_dir / "national_master_monthly.csv", index=False)
    print(f"Saved National Monthly data: {len(monthly_nat_df)} rows")

    # 2. Prepare District Data
    print("Preparing District data...")
    # Columns expected: Year, District, Male, Female, Total
    # Ensure columns match standard format
    district_df.rename(columns={'Total': 'District_Total', 'Male': 'District_Male', 'Female': 'District_Female'}, inplace=True)
    
    # We will interpolate district data into monthly data too, to match the national modeling standard
    district_monthly_data = []
    
    districts = district_df['District'].unique()
    for d in districts:
        d_df = district_df[district_df['District'] == d].sort_values('Year').reset_index(drop=True)
        if len(d_df) < 2:
            continue
            
        for i in range(len(d_df) - 1):
            year1_row = d_df.iloc[i]
            year2_row = d_df.iloc[i+1]
            y = year1_row["Year"]
            
            for m in months:
                frac = (m - 1) / 12.0
                interp_row = {"Year": y, "Month": m, "District": d}
                interp_row["Date"] = pd.to_datetime(f"{int(y)}-{int(m):02d}-01")
                
                for col in ["District_Male", "District_Female", "District_Total"]:
                    interp_row[col] = year1_row[col] + frac * (year2_row[col] - year1_row[col])
                district_monthly_data.append(interp_row)
                
        # Last year
        last_year_row = d_df.iloc[-1]
        prev_year_row = d_df.iloc[-2]
        y = last_year_row["Year"]
        for m in months:
            frac = (m - 1) / 12.0
            interp_row = {"Year": y, "Month": m, "District": d}
            interp_row["Date"] = pd.to_datetime(f"{int(y)}-{int(m):02d}-01")
            for col in ["District_Male", "District_Female", "District_Total"]:
                diff = last_year_row[col] - prev_year_row[col]
                interp_row[col] = last_year_row[col] + frac * diff
            district_monthly_data.append(interp_row)

    monthly_dist_df = pd.DataFrame(district_monthly_data)
    
    # Merge national features into district
    monthly_dist_df = monthly_dist_df.merge(
        monthly_nat_df.drop(columns=["Year", "Month"]), 
        on="Date", 
        how="left"
    )
    
    monthly_dist_df.to_csv(processed_dir / "district_master_monthly.csv", index=False)
    print(f"Saved District Monthly data: {len(monthly_dist_df)} rows")
    print("Done!")

if __name__ == "__main__":
    create_dataset()
