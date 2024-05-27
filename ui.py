"""
UI
"""
from nicegui import ui, run, app
from llm import recommend_products, get_product_details, instantiate_db, justify_recommendation

db = None

def on_startup():
    # Instantiatiate ChromaDB on startup, asynchronously
    global db
    db = instantiate_db()

AMAZON_PRODUCT_URL = "https://www.amazon.com/exec/obidos/ASIN/{ASIN}"

app.on_startup(on_startup)

@ui.page("/", title="Home")
def main():

    # This page sets up the search bar. 
    # User adds query and 'hardcoded' settings, then clicks search and is redirected to the results page
    with ui.button_group().classes("mt-64 mx-auto items-center justify-center"):
        query = ui.input(placeholder="Describe your dream...").classes("w-96 px-2")
        ui.button(icon="search", on_click=lambda: ui.navigate.to(f"/results?q={query.value}&min_rating={min_rating.value}&min_reviews={min_reviews.value}&max_price={max_price.value}") )

    ui.separator().classes("w-1/2 items-center justify-center mx-auto")

    # Add hardcoded settings here
    ui.markdown("#### Settings:").classes("mx-auto")
    with ui.grid(columns=3).classes("items-center justify-center mx-auto h-full"):
        min_rating = ui.number(label="Min. rating", min=0, max=5, step=0.5, value=4, suffix="★")
        min_reviews = ui.number(label="Min. # reviews", min=0, max=None, step=1, value=100)
        max_price = ui.number(label="Max price", min=1, max=1000, step=1, value=50, prefix="$")

@ui.page("/results", response_timeout=60)
async def results(q: str, min_rating: float = 0, min_reviews: int = 100, max_price: int = 50):

    def build_results(q, min_rating, min_reviews, max_price):
        results_list = []
        response = recommend_products(q, db=db, min_rating=min_rating, min_reviews=min_reviews, max_price=max_price).split(",")
        response = [x.strip() for x in response]
        for asin in response:
            product_data = get_product_details(asin, db=db)
            summary, positives, negatives, neutrals = justify_recommendation(q, product_data[0])
            summary_list = summary.split("\n-")
            positives_list = positives.split("\n-")
            negatives_list = negatives.split("\n-")
            neutrals_list = neutrals.split("\n-")
            results_list.append({
                "asin": asin,
                "product_data": product_data,
                "summary": summary_list,
                "positives": positives_list,
                "negatives": negatives_list,
                "neutrals": neutrals_list
            })
        return results_list

    # Show loading spinner while awaiting results.
    with ui.row() as loading_row:
        ui.spinner(type="bars", size="xl")
        ui.label("Loading results...")
    results_list = await run.io_bound(build_results, q, min_rating, min_reviews, max_price)
    loading_row.delete()

    # Display results
    ui.label("We think you'll like...").classes("text-2xl underline")
    with ui.carousel(arrows=True, navigation=True, animated=True).classes("w-full h-full").props("control-color=black control-type=unelevated"):
        for result in results_list:
            with ui.carousel_slide().classes("items-center justify-center w-full h-full"):
                results_card(result)


def results_card(result_data):
    with ui.card().classes("w-3/4 mb-12"):
        with ui.row().classes("w-full"):
            ui.label(result_data["product_data"][0]["title"]).classes("text-xl")
            ui.space()
            with ui.link("", target=AMAZON_PRODUCT_URL.format(ASIN=result_data["product_data"][0]["parent_asin"]), new_tab=True):
                ui.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg").classes("w-24")
                ui.tooltip("View on Amazon")
        with ui.row().classes("w-full"):
            ui.image(result_data["product_data"][0]["images"]).classes("w-48")
            with ui.column().classes("justify-right items-right"):
                ui.markdown("## **Price:** " + "$" + str(result_data["product_data"][0]["price"]))
                ui.markdown("**Rating:** " + str(result_data["product_data"][0]["average_rating"]) + " *(" + str(result_data["product_data"][0]["rating_number"]) + " reviews)*")
                for line in result_data["summary"][0:3]:
                    ui.chip(text=line, icon="info")
        ui.markdown("**Review summaries**")
        with ui.grid(columns=3):
            with ui.column():
                for line in result_data["positives"][0:3]:
                    ui.chip(text=line, color="green")
            with ui.column():
                for line in result_data["negatives"][0:3]:
                    ui.chip(text=line, color="red")
            with ui.column():
                for line in result_data["neutrals"][0:3]:
                    ui.chip(text=line, color="gray")

if __name__ in {"__main__", "__mp_main__"}:
    ui.run()