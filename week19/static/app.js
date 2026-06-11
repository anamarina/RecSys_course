const state = {
  pageSize: 5,
  shownItemIds: new Set(),
  dismissedItemIds: new Set(),
  cart: new Map(),
  feedItems: [],
  recommendations: [],
};

const feedGrid = document.getElementById("feedGrid");
const feedEmpty = document.getElementById("feedEmpty");
const cartList = document.getElementById("cartList");
const cartEmpty = document.getElementById("cartEmpty");
const recList = document.getElementById("recList");
const recEmpty = document.getElementById("recEmpty");
const loadMoreBtn = document.getElementById("loadMoreBtn");
const feedCounter = document.getElementById("feedCounter");
const cartCounter = document.getElementById("cartCounter");

function allExcludedIds() {
  const result = new Set([...state.shownItemIds, ...state.dismissedItemIds, ...state.cart.keys()]);
  return [...result];
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  const payload = await response.json();
  state.pageSize = payload.page_size || 5;
  await loadMoreFeed();
  renderAll();
}

async function loadMoreFeed() {
  loadMoreBtn.disabled = true;
  const params = new URLSearchParams();
  params.set("limit", String(state.pageSize));
  for (const itemId of allExcludedIds()) {
    params.append("exclude", itemId);
  }

  const response = await fetch(`/api/feed?${params.toString()}`);
  const payload = await response.json();
  const newItems = payload.items || [];
  newItems.forEach((item) => state.shownItemIds.add(item.item_id));
  state.feedItems = [...state.feedItems, ...newItems];
  loadMoreBtn.disabled = false;
  renderFeed();
}

async function refreshRecommendations() {
  if (state.cart.size === 0) {
    state.recommendations = [];
    renderRecommendations();
    return;
  }

  const response = await fetch("/api/recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      cart_item_ids: [...state.cart.keys()],
      exclude_item_ids: allExcludedIds(),
      limit: state.pageSize,
    }),
  });
  const payload = await response.json();
  state.recommendations = payload.items || [];
  renderRecommendations();
}

function addToCart(item) {
  const current = state.cart.get(item.item_id);
  if (current) {
    current.quantity += 1;
  } else {
    state.cart.set(item.item_id, { ...item, quantity: 1 });
  }
  removeFeedItem(item.item_id);
  renderCart();
  refreshRecommendations();
}

function decreaseFromCart(itemId) {
  const current = state.cart.get(itemId);
  if (!current) {
    return;
  }
  current.quantity -= 1;
  if (current.quantity <= 0) {
    state.cart.delete(itemId);
  }
  renderCart();
  refreshRecommendations();
}

function dismissItem(itemId) {
  state.dismissedItemIds.add(itemId);
  removeFeedItem(itemId);
  refreshRecommendations();
}

function removeFeedItem(itemId) {
  state.feedItems = state.feedItems.filter((item) => item.item_id !== itemId);
  renderFeed();
}

function renderAll() {
  renderFeed();
  renderCart();
  renderRecommendations();
}

function renderFeed() {
  feedGrid.innerHTML = "";
  state.feedItems.forEach((item) => {
    const card = document.createElement("article");
    card.className = "product-card";
    card.innerHTML = `
      <img src="${item.image_url}" alt="${item.item_name}">
      <div class="card-body">
        <span class="chip">${item.category}</span>
        <h3 class="title-line">${item.item_name}</h3>
        <p class="meta-line">Популярность в synthetic train baskets: ${item.popularity}</p>
        <div class="actions-row">
          <button class="btn btn-primary" data-action="add">В корзину</button>
          <button class="btn btn-secondary" data-action="dismiss">Скрыть</button>
        </div>
      </div>
    `;
    card.querySelector('[data-action="add"]').addEventListener("click", () => addToCart(item));
    card.querySelector('[data-action="dismiss"]').addEventListener("click", () => dismissItem(item.item_id));
    feedGrid.appendChild(card);
  });

  feedEmpty.classList.toggle("hidden", state.feedItems.length !== 0);
  feedCounter.textContent = String(state.shownItemIds.size);
}

function renderCart() {
  cartList.innerHTML = "";
  for (const item of state.cart.values()) {
    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <img src="${item.image_url}" alt="${item.item_name}">
      <div class="list-body">
        <h3 class="title-line">${item.item_name}</h3>
        <p class="meta-line">${item.category}</p>
        <div class="quantity-row">
          <span class="qty-badge">x${item.quantity}</span>
          <button class="btn btn-ghost" type="button">Убрать</button>
        </div>
      </div>
    `;
    card.querySelector("button").addEventListener("click", () => decreaseFromCart(item.item_id));
    cartList.appendChild(card);
  }

  cartCounter.textContent = String(state.cart.size);
  cartEmpty.classList.toggle("hidden", state.cart.size !== 0);
}

function renderRecommendations() {
  recList.innerHTML = "";
  state.recommendations.forEach((item) => {
    const reasonText = item.reasons && item.reasons.length
      ? `Похоже на покупки вместе с: ${item.reasons.join(", ")}`
      : "Fallback на популярные товары";

    const card = document.createElement("article");
    card.className = "list-card";
    card.innerHTML = `
      <img src="${item.image_url}" alt="${item.item_name}">
      <div class="list-body">
        <h3 class="title-line">${item.item_name}</h3>
        <p class="meta-line">${item.category}</p>
        <p class="reason-line">${reasonText}</p>
        <div class="actions-row">
          <span class="qty-badge">score ${Number(item.score || 0).toFixed(2)}</span>
          <button class="btn btn-primary" type="button">Добавить</button>
        </div>
      </div>
    `;
    card.querySelector("button").addEventListener("click", () => addToCart(item));
    recList.appendChild(card);
  });

  recEmpty.classList.toggle("hidden", state.recommendations.length !== 0);
}

loadMoreBtn.addEventListener("click", loadMoreFeed);
bootstrap();
