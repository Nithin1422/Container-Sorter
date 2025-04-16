
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_optimization(df):
    # ✅ Ensure Manual Load is numeric and filled
    df["Manual Load"] = pd.to_numeric(df["Manual Load"], errors="coerce").fillna(0)

    # ➕ Calculate used space from manual input
    df["Used Volume"] = df["Manual Load"] * df["Volume (cbm)"]
    df["Used Weight"] = df["Manual Load"] * df["Weight (kg)"]
    used_volume_total = df["Used Volume"].sum()
    used_weight_total = df["Used Weight"].sum()

    # 🔁 Generate synthetic training data for ML
    simulated_data = []
    np.random.seed(42)
    for _ in range(1000):
        for _, row in df.iterrows():
            rem_volume = np.random.uniform(5, 67)
            rem_weight = np.random.uniform(5000, 28000)
            usage = row["Weekly Usage"] + np.random.randint(-3, 4)
            stock = row["Stock on Hand"] + np.random.randint(-50, 50)
            cover = stock / (usage + 1e-5)
            priority = ((usage / (cover + 1e-2)) * 10 +
                        (1 / (cover + 1.1)) * 100 +
                        (1 / (row["Lead Time"] + 1e-2)) * 50)

            max_cartons_by_volume = int(rem_volume // row["Volume (cbm)"])
            max_cartons_by_weight = int(rem_weight // row["Weight (kg)"])
            realistic_max = min(max_cartons_by_volume, max_cartons_by_weight, stock)
            cartons_to_load = max(0, int(realistic_max * np.random.uniform(0.5, 1.0)))

            simulated_data.append([
                rem_volume, rem_weight, row["Volume (cbm)"], row["Weight (kg)"],
                usage, stock, cover, row["Stock on Order"], row["Lead Time"],
                row["MOQ"], row["Safety Stock"], row["ROP"], priority,
                cartons_to_load
            ])

    sim_df = pd.DataFrame(simulated_data, columns=[
        "Remaining Volume", "Remaining Weight", "SKU Volume", "SKU Weight",
        "Weekly Usage", "Stock on Hand", "Weeks of Cover", "Stock on Order", "Lead Time",
        "MOQ", "Safety Stock", "ROP", "Priority Score", "Cartons"
    ])

    # ⚙️ Train Random Forest Regressor
    X = sim_df.drop(columns=["Cartons"])
    y = sim_df["Cartons"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 🚚 Hybrid fill for 100% container utilization
    def hybrid_fill_full(container_name, max_volume, max_weight):
        remaining_volume = max_volume - used_volume_total
        remaining_weight = max_weight - used_weight_total

        predictions = []
        for _, row in df[df["Manual Load"].fillna(0) == 0].iterrows():
            weeks_of_cover = row["Stock on Hand"] / (row["Weekly Usage"] + 1e-5)
            input_data = pd.DataFrame([{
                "Remaining Volume": remaining_volume,
                "Remaining Weight": remaining_weight,
                "SKU Volume": row["Volume (cbm)"],
                "SKU Weight": row["Weight (kg)"],
                "Weekly Usage": row["Weekly Usage"],
                "Stock on Hand": row["Stock on Hand"],
                "Weeks of Cover": weeks_of_cover,
                "Stock on Order": row["Stock on Order"],
                "Lead Time": row["Lead Time"],
                "MOQ": row["MOQ"],
                "Safety Stock": row["Safety Stock"],
                "ROP": row["ROP"],
                "Priority Score": (row["Weekly Usage"] / (weeks_of_cover + 1e-2)) * 10 +
                                  (1 / (weeks_of_cover + 1.1)) * 100 +
                                  (1 / (row["Lead Time"] + 1e-2)) * 50
            }])
            input_scaled = scaler.transform(input_data)
            predicted_qty = int(model.predict(input_scaled)[0] * 1.8)

            max_qty_by_volume = int(remaining_volume // row["Volume (cbm)"])
            max_qty_by_weight = int(remaining_weight // row["Weight (kg)"])
            max_possible = min(predicted_qty, max_qty_by_volume, max_qty_by_weight,
                               row["Stock on Hand"] + row["Stock on Order"])

            if max_possible >= row["MOQ"] and max_possible > row["Safety Stock"]:
                predictions.append({
                    "Container Type": container_name,
                    "SKU": row["SKU Code"],
                    "Predicted Qty": max_possible,
                    "Used Volume": round(max_possible * row["Volume (cbm)"], 3),
                    "Used Weight": round(max_possible * row["Weight (kg)"], 2)
                })

        # ✅ Fix: Check for empty predictions
        pred_df = pd.DataFrame(predictions)
        if pred_df.empty:
            return container_name, 0, 0, pd.DataFrame()

        pred_df = pred_df.sort_values(by="Used Volume").reset_index(drop=True)
        final_selection, vol_accum, wt_accum = [], 0, 0

        for _, row in pred_df.iterrows():
            if vol_accum + row["Used Volume"] <= remaining_volume and wt_accum + row["Used Weight"] <= remaining_weight:
                final_selection.append(row)
                vol_accum += row["Used Volume"]
                wt_accum += row["Used Weight"]
            else:
                break

        return container_name, vol_accum, wt_accum, pd.DataFrame(final_selection)

    # 📦 Run for both container types
    result_20ft = hybrid_fill_full("20ft", 33, 28080)
    result_40ft = hybrid_fill_full("40ft", 67, 26700)

    # 🧮 Calculate fill scores
    fill_20ft_vol = result_20ft[1] / 33
    fill_20ft_wt = result_20ft[2] / 28080
    fill_40ft_vol = result_40ft[1] / 67
    fill_40ft_wt = result_40ft[2] / 26700

    score_20ft = max(fill_20ft_vol, fill_20ft_wt)
    score_40ft = max(fill_40ft_vol, fill_40ft_wt)

    # 🥇 Pick best result
    best_result = result_40ft if score_40ft >= score_20ft else result_20ft

    # ➕ Add manual load to output
    manual_df = df[df["Manual Load"] > 0][["SKU Code", "Manual Load"]].copy()
    manual_df.columns = ["SKU", "Predicted Qty"]
    manual_df["Used Volume"] = manual_df["SKU"].map(df.set_index("SKU Code")["Volume (cbm)"]) * manual_df["Predicted Qty"]
    manual_df["Used Weight"] = manual_df["SKU"].map(df.set_index("SKU Code")["Weight (kg)"]) * manual_df["Predicted Qty"]
    manual_df["Container Type"] = best_result[0]

    final_df = pd.concat([manual_df, best_result[3]], ignore_index=True)

    summary_df = pd.DataFrame({
        "Metric": ["Volume Fill %", "Weight Fill %"],
        "20ft": [fill_20ft_vol * 100, fill_20ft_wt * 100],
        "40ft": [fill_40ft_vol * 100, fill_40ft_wt * 100]
    })

    return best_result[0], final_df, summary_df
