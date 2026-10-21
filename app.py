import streamlit as st

from src.predict import load_bundle, predict_price

st.set_page_config(page_title="India House Price Predictor", page_icon="🏠")


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()

st.title("🏠 India House Price Predictor")
st.write("Estimate a property's price from a few details. "
         "Built with a Random Forest trained on ~28,500 Indian listings.")

col1, col2 = st.columns(2)
with col1:
    cities = sorted(c for c in bundle["cities"] if c != "Other") + ["Other"]
    city = st.selectbox(
        "City", cities,
        index=cities.index("Bangalore") if "Bangalore" in cities else 0,
        format_func=lambda c: "Other / smaller city (rough estimate)" if c == "Other" else c,
    )
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6], index=1)
    sqft = st.number_input("Area (sq ft)", min_value=200, max_value=10000, value=1000, step=50)
with col2:
    posted_by = st.selectbox("Posted by", ["Owner", "Dealer", "Builder"])
    under_construction = st.checkbox("Under construction")
    rera = st.checkbox("RERA approved")
    resale = st.checkbox("Resale property", value=True)

if city == "Other":
    st.warning("'Other' groups many smaller cities and uses an average central-India location, "
               "so this estimate is very rough. Pick a named city for a better result.")

if st.button("Predict price", type="primary"):
    price = predict_price(
        bundle, bhk=bhk, sqft=sqft, city=city, posted_by=posted_by,
        under_construction=int(under_construction), rera=int(rera),
        ready_to_move=int(not under_construction), resale=int(resale),
    )
    st.metric("Estimated price", f"₹ {price:,.1f} lakhs")
    st.caption(f"≈ ₹ {price / 100:,.2f} crore")

    err = bundle["metrics"]["median_pct_error"]
    st.caption(
        f"On unseen listings the model's typical error was about {err:.0f}%, "
        "and it is less reliable for expensive homes. The location used is the city's "
        "average position, so this is a city-level estimate, not a street-level one."
    )