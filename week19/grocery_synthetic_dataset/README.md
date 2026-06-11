# Synthetic Grocery Baskets Dataset — 300 orders

Данные полностью синтетические, но корзины собраны через тематические шаблоны, чтобы заказы выглядели логично: завтрак, борщ, паста, салаты, недельная закупка, детские товары, зоотовары, бытовая закупка, суши, карри, бургеры, пикник и т.д.

## Файлы

- `grocery_baskets_line_items.csv` — основной датасет: одна строка = один товар в заказе.
- `grocery_baskets_minimal.csv` — минимальный формат: `basket_id, item_name`.
- `item_catalog.csv` — справочник товаров: `item_id, item_name, category, image_path`.
- `grocery_baskets_summary.csv` — одна строка = одна корзина, товары перечислены через `|`.
- `item_images/` — отдельная PNG-картинка для каждого `item_id`.
- `dataset_manifest.json` — описание схемы и статистика.

## Основная схема line-items

`basket_id, item_id, item_name, category, quantity, basket_theme`

Поле `basket_theme` можно использовать для анализа или удалить, если нужен только `basket_id → item`.
