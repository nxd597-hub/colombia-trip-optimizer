import streamlit as st
import pandas as pd
import pulp as pl

# ---------------------------
# Helper: parse "1 to 7" style availability
# ---------------------------
def parse_days(avail_str, all_days):
    """Return list of days where item is available."""
    if isinstance(avail_str, str) and "to" in avail_str:
        parts = avail_str.split("to")
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            return [d for d in all_days if start <= d <= end]
        except:
            return all_days
    else:
        # If blank or malformed, assume all days
        return all_days


# ---------------------------
# Core solver
# ---------------------------
def solve_itinerary(travel, hotels, activities,
                    DAYS=7, BUDGET=2000,
                    MIN_ACTS=2, HMAX=6,
                    MAX_NIGHTS_PER_CITY=4,
                    require_cartagena=True):

    days = list(range(1, DAYS + 1))

    # --- Sets from data ---
    cities = sorted(hotels["City"].unique())

    # hotels by city
    hotels_by_city = {
        c: hotels.loc[hotels["City"] == c, "Hotel"].tolist()
        for c in cities
    }

    # activities by city
    acts_by_city = {
        c: activities.loc[activities["City"] == c, "Activity"].tolist()
        for c in cities
    }

    # travel arcs (i,j,m)
    modes = sorted(travel["Mode"].unique())
    travel_arcs = []
    for _, row in travel.iterrows():
        i = row["From"]
        j = row["To"]
        m = row["Mode"]
        travel_arcs.append((i, j, m))

    # cost and time dictionaries
    travel_cost = {}
    travel_time = {}
    travel_avail = {}
    for _, row in travel.iterrows():
        key = (row["From"], row["To"], row["Mode"])
        travel_cost[key] = float(row["Cost"])
        travel_time[key] = float(row["Time_hours"])
        travel_avail[key] = parse_days(row.get("Avail_Days", ""), days)

    hotel_cost = {}
    hotel_avail = {}
    for _, row in hotels.iterrows():
        key = (row["City"], row["Hotel"])
        hotel_cost[key] = float(row["NightCost"])
        hotel_avail[key] = parse_days(row.get("Avail_Days", ""), days)

    act_cost = {}
    act_dur = {}
    act_avail = {}
    for _, row in activities.iterrows():
        key = (row["City"], row["Activity"])
        act_cost[key] = float(row["Cost"])
        act_dur[key] = float(row["Time_hours"])
        act_avail[key] = parse_days(row.get("Avail_Days", ""), days)

    # ---------------------------
    # Build MILP model
    # ---------------------------
    mdl = pl.LpProblem("Colombia_Trip", pl.LpMinimize)

    # Decision variables
    x = pl.LpVariable.dicts("x", (cities, days), 0, 1, pl.LpBinary)
    h = pl.LpVariable.dicts(
        "h", ((c, hn, d) for c in cities for hn in hotels_by_city[c] for d in days),
        0, 1, pl.LpBinary
    )
    a = pl.LpVariable.dicts(
        "a", ((c, an, d) for c in cities for an in acts_by_city[c] for d in days),
        0, 1, pl.LpBinary
    )
    t = pl.LpVariable.dicts(
        "t", ((i, j, m, d) for (i, j, m) in travel_arcs for d in days[:-1]),
        0, 1, pl.LpBinary
    )

    # Objective: minimize total cost
    mdl += (
        pl.lpSum(
            travel_cost[(i, j, m)] * t[(i, j, m, d)]
            for (i, j, m) in travel_arcs for d in days[:-1]
        )
        + pl.lpSum(
            hotel_cost[(c, hn)] * h[(c, hn, d)]
            for c in cities for hn in hotels_by_city[c] for d in days
        )
        + pl.lpSum(
            act_cost[(c, an)] * a[(c, an, d)]
            for c in cities for an in acts_by_city[c] for d in days
        )
    )

    # 1) One city per day
    for d in days:
        mdl += pl.lpSum(x[c][d] for c in cities) == 1

    # 2) Hotel per city/day (and availability)
    for c in cities:
        for d in days:
            mdl += (
                pl.lpSum(h[(c, hn, d)] for hn in hotels_by_city[c]) == x[c][d]
            )
            # hotel availability
            for hn in hotels_by_city[c]:
                if d not in hotel_avail[(c, hn)]:
                    mdl += h[(c, hn, d)] == 0

    # 3) Activities only if in city, within hours, and available
    for c in cities:
        for d in days:
            # link to being in city
            for an in acts_by_city[c]:
                mdl += a[(c, an, d)] <= x[c][d]
                if d not in act_avail[(c, an)]:
                    mdl += a[(c, an, d)] == 0

            # daily hours
            mdl += pl.lpSum(
                act_dur[(c, an)] * a[(c, an, d)]
                for an in acts_by_city[c]
            ) <= HMAX

    # 4) Minimum number of activities
    mdl += pl.lpSum(
        a[(c, an, d)] for c in cities for an in acts_by_city[c] for d in days
    ) >= MIN_ACTS

    # 5) Travel arcs: availability
    for (i, j, m) in travel_arcs:
        for d in days[:-1]:
            if d not in travel_avail[(i, j, m)]:
                mdl += t[(i, j, m, d)] == 0

    # 6) Flow conservation
    for j in cities:
        for d in days[:-1]:
            inflow = pl.lpSum(
                t[(i, j, m, d)] for (i, jj, m) in travel_arcs if jj == j
            )
            outflow = pl.lpSum(
                t[(j, k, m, d)] for (jj, k, m) in travel_arcs if jj == j
            )
            mdl += x[j][d + 1] == x[j][d] + inflow - outflow

    # 7) Start in Bogota (if present)
    if "Bogota" in cities:
        mdl += x["Bogota"][1] == 1

    # 8) Must visit Cartagena at least once (optional)
    if require_cartagena and "Cartagena" in cities:
        mdl += pl.lpSum(x["Cartagena"][d] for d in days) >= 1

    # 9) Max nights per city
    for c in cities:
        mdl += pl.lpSum(x[c][d] for d in days) <= MAX_NIGHTS_PER_CITY

    # 10) Budget constraint
    mdl += (
        pl.lpSum(
            travel_cost[(i, j, m)] * t[(i, j, m, d)]
            for (i, j, m) in travel_arcs for d in days[:-1]
        )
        + pl.lpSum(
            hotel_cost[(c, hn)] * h[(c, hn, d)]
            for c in cities for hn in hotels_by_city[c] for d in days
        )
        + pl.lpSum(
            act_cost[(c, an)] * a[(c, an, d)]
            for c in cities for an in acts_by_city[c] for d in days
        )
        <= BUDGET
    )

    # ---------------------------
    # Solve
    # ---------------------------
    mdl.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[mdl.status]

    if status != "Optimal":
        return {
            "status": status,
            "itinerary": None,
            "travel_cost": None,
            "hotel_cost": None,
            "activity_cost": None,
            "num_transfers": None,
            "travel_hours": None,
        }

    # ---------------------------
    # Build itinerary table
    # ---------------------------
    rows = []
    for d in days:
        # city chosen
        city = next(c for c in cities if pl.value(x[c][d]) > 0.5)

        # hotel chosen
        hotel_name = None
        night_cost = 0.0
        for hn in hotels_by_city[city]:
            if pl.value(h[(city, hn, d)]) > 0.5:
                hotel_name = hn
                night_cost = hotel_cost[(city, hn)]
                break

        # activities
        act_list = []
        for an in acts_by_city[city]:
            if pl.value(a[(city, an, d)]) > 0.5:
                act_list.append(f"{an} ({act_dur[(city, an)]}h, ${act_cost[(city, an)]})")

        rows.append(
            {
                "Day": d,
                "City": city,
                "Hotel": hotel_name,
                "NightCost": night_cost,
                "Activities": ", ".join(act_list) if act_list else "—",
            }
        )

    itinerary_df = pd.DataFrame(rows)

    # Costs
    total_travel_cost = sum(
        travel_cost[(i, j, m)] * pl.value(t[(i, j, m, d)])
        for (i, j, m) in travel_arcs for d in days[:-1]
    )
    total_hotel_cost = sum(
        hotel_cost[(c, hn)] * pl.value(h[(c, hn, d)])
        for c in cities for hn in hotels_by_city[c] for d in days
    )
    total_activity_cost = sum(
        act_cost[(c, an)] * pl.value(a[(c, an, d)])
        for c in cities for an in acts_by_city[c] for d in days
    )

    # Transfers and hours
    num_transfers = int(
        sum(pl.value(t[(i, j, m, d)]) for (i, j, m) in travel_arcs for d in days[:-1])
    )
    total_travel_hours = sum(
        travel_time[(i, j, m)] * pl.value(t[(i, j, m, d)])
        for (i, j, m) in travel_arcs for d in days[:-1]
    )

    return {
        "status": status,
        "itinerary": itinerary_df,
        "travel_cost": total_travel_cost,
        "hotel_cost": total_hotel_cost,
        "activity_cost": total_activity_cost,
        "num_transfers": num_transfers,
        "travel_hours": total_travel_hours,
    }


# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="Colombia Trip Optimization", layout="wide")
    st.title("Colombia Trip Optimization Dashboard")

    st.markdown(
        "This dashboard runs a mixed-integer optimization model to build an "
        "optimal multi-city trip in Colombia based on your inputs."
    )

    st.sidebar.header("Inputs")

    # File uploads
    st.sidebar.subheader("Upload Data (CSV)")
    travel_file = st.sidebar.file_uploader("Travel data", type=["csv"])
    hotels_file = st.sidebar.file_uploader("Hotels data", type=["csv"])
    acts_file = st.sidebar.file_uploader("Activities data", type=["csv"])

    st.sidebar.markdown("**Model Parameters**")
    DAYS = st.sidebar.slider("Number of days", min_value=3, max_value=14, value=7, step=1)
    BUDGET = st.sidebar.number_input("Budget ($)", min_value=200, max_value=20000, value=2000, step=50)
    MIN_ACTS = st.sidebar.slider("Minimum number of activities", 0, 10, 2)
    HMAX = st.sidebar.slider("Max activity hours per day", 1, 12, 6)
    MAX_NIGHTS_PER_CITY = st.sidebar.slider("Max nights per city", 1, DAYS, 4)
    require_cartagena = st.sidebar.checkbox("Require visiting Cartagena at least once?", value=True)

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Optimization")

    if not run_button:
        st.info("Adjust inputs on the left, then click **Run Optimization**.")
        return

    # Load data
    if not (travel_file and hotels_file and acts_file):
        st.error("Please upload all three CSV files.")
        return

    travel = pd.read_csv(travel_file)
    hotels = pd.read_csv(hotels_file)
    activities = pd.read_csv(acts_file)

    with st.spinner("Solving optimization model..."):
        result = solve_itinerary(
            travel, hotels, activities,
            DAYS=DAYS,
            BUDGET=BUDGET,
            MIN_ACTS=MIN_ACTS,
            HMAX=HMAX,
            MAX_NIGHTS_PER_CITY=MAX_NIGHTS_PER_CITY,
            require_cartagena=require_cartagena,
        )

    status = result["status"]
    st.subheader("Solver Status")
    st.write(status)

    if status != "Optimal":
        st.error("No feasible itinerary found for this combination of parameters.")
        return

    itinerary_df = result["itinerary"]
    travel_cost = result["travel_cost"]
    hotel_cost = result["hotel_cost"]
    activity_cost = result["activity_cost"]
    num_transfers = result["num_transfers"]
    total_travel_hours = result["travel_hours"]
    total_cost = travel_cost + hotel_cost + activity_cost

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cost", f"${total_cost:,.0f}")
    col2.metric("Travel Cost", f"${travel_cost:,.0f}")
    col3.metric("Hotel Cost", f"${hotel_cost:,.0f}")
    col4.metric("Activity Cost", f"${activity_cost:,.0f}")

    col5, col6 = st.columns(2)
    col5.metric("Transfers", f"{num_transfers}")
    col6.metric("Travel Hours", f"{total_travel_hours:.1f}")

    st.subheader("Optimized Itinerary")
    st.dataframe(itinerary_df, use_container_width=True)

    # Charts
    st.subheader("Cost Breakdown")
    cost_df = pd.DataFrame(
        {
            "Component": ["Travel", "Hotel", "Activities"],
            "Cost": [travel_cost, hotel_cost, activity_cost],
        }
    )
    st.bar_chart(cost_df.set_index("Component"))

    st.subheader("Nights per City")
    nights_df = itinerary_df.groupby("City")["Day"].count().reset_index()
    nights_df = nights_df.rename(columns={"Day": "Nights"})
    st.bar_chart(nights_df.set_index("City"))

    st.download_button(
        label="Download Itinerary as CSV",
        data=itinerary_df.to_csv(index=False),
        file_name="itinerary.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
