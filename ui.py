"""
UI
"""
from nicegui import ui, run
from llm import recommend_products, get_product_details, instantiate_db, justify_recommendation

# Instantiatiate ChromaDB on startup, asynchronously
db = instantiate_db()

AMAZON_PRODUCT_URL = "https://www.amazon.com/exec/obidos/ASIN/{ASIN}"

@ui.page("/")
def main():

    # This page sets up the search bar. User adds query and 'hardcoded' settings, then clicks search and is redirected to the results page
    with ui.button_group():
        query = ui.input(placeholder="What do you want from a giftcard?").classes("w-96")
        ui.button(icon="search", on_click=lambda: ui.navigate.to(f"/results?q={query.value}") )

    ui.separator()

    # Add hardcoded settings here
    ui.label("Settings:")
    min_rating = ui.number(label="Minimum rating", min=0, max=5, step=0.5, value=0)
    ui.label("Price range")
    price_range = ui.range(min=0, max=100, step=1, value=[0, 100])

@ui.page("/results")
def results(q: str):

    # Show loading spinner while awaiting results.
    results_dict = {}
    loading = ui.spinner()
    response = recommend_products(q, db=db).split(",")
    response = [x.strip() for x in response]
    top_asin = response[0]
    for asin in response:
        product_data = get_product_details(asin, db=db)
        summary, positives, negatives, neutrals = justify_recommendation(q, product_data[0])
        summary_list = summary.split("\n-")
        positives_list = positives.split("\n-")
        negatives_list = negatives.split("\n-")
        neutrals_list = neutrals.split("\n-")
        results_dict[asin] = {
            "product_data": product_data,
            "summary": summary_list,
            "positives": positives_list,
            "negatives": negatives_list,
            "neutrals": neutrals_list
        }
    loading.delete()

    # Display results
    ui.label("Top result")
    with ui.card():
        top_result = results_dict[top_asin]
        ui.label(top_result["product_data"][0]["title"])
        with ui.row():
            ui.image(top_result["product_data"][0]["images"]).classes("w-48")
            with ui.column():
                ui.label("Rating: " + str(top_result["product_data"][0]["average_rating"]) + " (" + str(top_result["product_data"][0]["rating_number"]) + " reviews)")
                ui.label("Price: " + "$" + str(top_result["product_data"][0]["price"]))
                for line in top_result["summary"][1:4]:
                    ui.chip(text=line, icon="info")
                ui.link("View on Amazon", target=AMAZON_PRODUCT_URL.format(ASIN=top_result["product_data"][0]["parent_asin"]), new_tab=True)
        with ui.row():
            with ui.column():
                for line in top_result["positives"][1:4]:
                    ui.chip(text=line, color="green")
            with ui.column():
                for line in top_result["negatives"][1:4]:
                    ui.chip(text=line, color="red")
            with ui.column():
                for line in top_result["neutrals"][1:4]:
                    ui.chip(text=line, color="gray")

if __name__ in {"__main__", "__mp_main__"}:
    ui.run()