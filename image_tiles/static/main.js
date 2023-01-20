function set_items_per_row(items_per_row) {
    console.log(items_per_row)
    items_per_row = parseInt(items_per_row)

    var margin = (1 - 0.01*items_per_row);
    var item_percentage = (100 / items_per_row) * margin;

    lis = document.getElementsByTagName("li");

    for (const item of lis) {
        item.style.width = item_percentage.toString() + "%";
    }
}

var url = new URL(window.location);
var items_per_row = url.searchParams.get("items_per_row");
set_items_per_row(items_per_row);
