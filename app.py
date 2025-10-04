import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
from typing import List, Tuple, Dict

# -------------------------------
# NumPy-based MLP implementation
# -------------------------------

Activation = str

# ReLU (Rectified Linear Unit)
def relu(x: np.ndarray) -> np.ndarray:
	return np.maximum(0.0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
	return (x > 0.0).astype(x.dtype)

# Tanh (Hyperbolic Tangent)
def tanh(x: np.ndarray) -> np.ndarray:
	return np.tanh(x)


def tanh_deriv(x: np.ndarray) -> np.ndarray:
	y = np.tanh(x)
	return 1.0 - y * y

# Logistic (Sigmoid)
def logistic(x: np.ndarray) -> np.ndarray:
	# numerically stable sigmoid
	out = np.empty_like(x)
	pos = x >= 0
	neg = ~pos
	out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
	ex = np.exp(x[neg])
	out[neg] = ex / (1.0 + ex)
	return out


def logistic_deriv(x: np.ndarray) -> np.ndarray:
	s = logistic(x)
	return s * (1.0 - s)


ACT_FUNCS = {
	"relu": (relu, relu_deriv),
	"tanh": (tanh, tanh_deriv),
	"logistic": (logistic, logistic_deriv),
}


class MLP:
	def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int,
				 activation: Activation = "relu", seed: int = 42):
		if activation not in ACT_FUNCS:
			raise ValueError(f"Unknown activation: {activation}")
		self.act, self.act_deriv = ACT_FUNCS[activation]
		self.layers = [input_dim] + hidden_layers + [output_dim]
		self.rng = np.random.default_rng(seed)
		self.params = self._init_params()

	def _init_params(self) -> Dict[str, np.ndarray]:
		params = {}
		for i in range(len(self.layers) - 1):
			fan_in = self.layers[i]
			fan_out = self.layers[i + 1]
			# He init for relu, Xavier for others
			if self.act is relu:
				scale = np.sqrt(2.0 / fan_in)
			else:
				scale = np.sqrt(1.0 / fan_in)
			W = self.rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64)
			b = np.zeros((1, fan_out), dtype=np.float64)
			params[f"W{i}"] = W
			params[f"b{i}"] = b
		return params

	def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		a_list = [X]
		z_list = []
		for i in range(len(self.layers) - 2):
			W = self.params[f"W{i}"]
			b = self.params[f"b{i}"]
			z = a_list[-1] @ W + b
			a = self.act(z)
			z_list.append(z)
			a_list.append(a)
		# output layer (logits)
		W = self.params[f"W{len(self.layers) - 2}"]
		b = self.params[f"b{len(self.layers) - 2}"]
		logits = a_list[-1] @ W + b
		z_list.append(logits)
		a_list.append(logits)
		return a_list, z_list

	@staticmethod
	def softmax(logits: np.ndarray) -> np.ndarray:
		# stable softmax
		shifted = logits - logits.max(axis=1, keepdims=True)
		exps = np.exp(shifted)
		return exps / (exps.sum(axis=1, keepdims=True) + 1e-12)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		a_list, _ = self.forward(X)
		logits = a_list[-1]
		return self.softmax(logits)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return np.argmax(self.predict_proba(X), axis=1)

	def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 200,
			batch_size: int = 32, l2: float = 0.0, verbose: bool = False):
		n_samples, _ = X.shape
		n_classes = self.layers[-1]
		y_onehot = np.eye(n_classes, dtype=np.float64)[y]

		for epoch in range(epochs):
			# shuffle
			idx = np.random.permutation(n_samples)
			Xs = X[idx]
			ys = y_onehot[idx]
			for start in range(0, n_samples, batch_size):
				end = min(start + batch_size, n_samples)
				xb = Xs[start:end]
				yb = ys[start:end]

				# forward
				a_list, z_list = self.forward(xb)
				logits = a_list[-1]
				probs = self.softmax(logits)

				# loss gradient (cross-entropy with softmax)
				grad_logits = (probs - yb) / xb.shape[0]

				# backprop
				grads_W = {}
				grads_b = {}

				# output layer grads
				last_i = len(self.layers) - 2
				grads_W[f"W{last_i}"] = a_list[-2].T @ grad_logits + l2 * self.params[f"W{last_i}"]
				grads_b[f"b{last_i}"] = grad_logits.sum(axis=0, keepdims=True)

				grad_a = grad_logits @ self.params[f"W{last_i}"].T

				# hidden layers in reverse
				for i in reversed(range(len(self.layers) - 2)):
					z = z_list[i]
					da_dz = self.act_deriv(z)
					grad_z = grad_a * da_dz
					grads_W[f"W{i}"] = a_list[i].T @ grad_z + l2 * self.params[f"W{i}"]
					grads_b[f"b{i}"] = grad_z.sum(axis=0, keepdims=True)
					if i > 0:
						grad_a = grad_z @ self.params[f"W{i}"].T
					# for i == 0, no need to compute grad for previous a (input layer)

				# update
				for k in self.params:
					if k.startswith('W'):
						self.params[k] -= lr * grads_W[k]
					else:
						self.params[k] -= lr * grads_b[k]

			if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
				preds = self.predict(X)
				acc = (preds == y).mean() if n_samples > 0 else 0.0
				print(f"Epoch {epoch+1}/{epochs} - acc: {acc:.3f}")


# -------------------------------
# Interactive UI
# -------------------------------

class InteractiveMLP:
	def __init__(self):
		self.points: List[Tuple[float, float]] = []
		self.labels: List[int] = []
		self.class_names: List[str] = ["Class 0", "Class 1"]
		self.class_colors: List[str] = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
		self.selected_class: int = 0

		# model config
		self.activation: Activation = "relu"
		self.hidden_layers: List[int] = [5, 3]
		self.lr: float = 0.1
		self.epochs: int = 200
		self.batch_size: int = 32
		self.l2: float = 0.0

		self.model: MLP | None = None

		self._build_ui()
		self._update_plot()

	def _build_ui(self):
		self.fig = plt.figure(figsize=(9, 6))
		# main axes for scatter and boundary (revert to 0.6 width)
		self.ax = self.fig.add_axes([0.06, 0.1, 0.6, 0.8])
		self.ax.set_title("Interactive MLP Decision Boundary")
		self.ax.set_xlim(-5, 5)
		self.ax.set_ylim(-5, 5)
		self.ax.grid(True, alpha=1)

		# side panel for controls
		self.control_ax = self.fig.add_axes([0.7, 0.1, 0.28, 0.8])
		self.control_ax.axis('off')

		# -------- Unified grid layout in the control panel --------
		# revert control panel to original placement
		panel_left, panel_bottom, panel_width, panel_height = 0.7, 0.1, 0.28, 0.8
		def panel_rect(rx: float, ry: float, rw: float, rh: float):
			return [panel_left + rx * panel_width,
					panel_bottom + ry * panel_height,
					rw * panel_width,
					rh * panel_height]

		pad_x = 0.02
		row_y = 1.0

		# Row 1: Class radio
		row_h = 0.18
		row_y -= row_h
		self.ax_class = self.fig.add_axes(panel_rect(pad_x, row_y, 1.0 - 2 * pad_x, row_h))
		self.radio_class = RadioButtons(self.ax_class, labels=[n for n in self.class_names], active=0)
		self.radio_class.on_clicked(self._on_class_change)

		# Row 2: +Class / -Class buttons (same row, two columns)
		row_h = 0.06
		row_y -= (row_h + 0.01)
		self.ax_add_class = self.fig.add_axes(panel_rect(pad_x, row_y, 0.48 - pad_x, row_h))
		self.btn_add_class = Button(self.ax_add_class, "+ Class")
		self.btn_add_class.on_clicked(self._on_add_class)
		self.ax_del_class = self.fig.add_axes(panel_rect(0.52, row_y, 0.48 - pad_x, row_h))
		self.btn_del_class = Button(self.ax_del_class, "- Class")
		self.btn_del_class.on_clicked(self._on_del_class)

		# Row 3: Activation radio
		row_h = 0.14
		row_y -= (row_h + 0.02)
		self.ax_act = self.fig.add_axes(panel_rect(pad_x, row_y, 1.0 - 2 * pad_x, row_h))
		self.radio_act = RadioButtons(self.ax_act, labels=["relu", "tanh", "logistic"], active=0)
		self.radio_act.on_clicked(self._on_act_change)

		# Rows 4-8: TextBoxes (Layers, LR, Epochs, Batch, L2)
		self.input_labels: Dict[str, any] = {}
		def add_textbox(label: str, initial: str, key: str):
			# Draw a non-editable label inside the textbox axes; textbox value remains just the input
			ax = self.fig.add_axes(panel_rect(pad_x, row_y, 1.0 - 2 * pad_x, 0.06))
			box = TextBox(ax, "", initial=initial)
			lbl = ax.text(0.02, 0.5, label, transform=ax.transAxes, va='center', ha='left', color='gray', zorder=10)
			self.input_labels[key] = lbl
			# Shift input text to the right so it doesn't overlap with the label
			try:
				box.text_disp.set_ha('left')
				box.text_disp.set_va('center')
				box.text_disp.set_position((0.3, 0.5))
			except Exception:
				pass
			return ax, box

		row_y -= (0.06 + 0.02)
		self.ax_layers, self.tbx_layers = add_textbox("Layers:", "5,3", "layers")
		self.tbx_layers.on_submit(self._on_layers_change)

		row_y -= (0.06 + 0.02)
		self.ax_lr, self.tbx_lr = add_textbox("LR:", str(self.lr), "lr")
		self.tbx_lr.on_submit(self._on_params_change)

		row_y -= (0.06 + 0.02)
		self.ax_epochs, self.tbx_epochs = add_textbox("Epochs:", str(self.epochs), "epochs")
		self.tbx_epochs.on_submit(self._on_params_change)

		row_y -= (0.06 + 0.02)
		self.ax_batch, self.tbx_batch = add_textbox("Batch:", str(self.batch_size), "batch")
		self.tbx_batch.on_submit(self._on_params_change)

		row_y -= (0.06 + 0.02)
		self.ax_l2, self.tbx_l2 = add_textbox("L2:", str(self.l2), "l2")
		self.tbx_l2.on_submit(self._on_params_change)

		# Row 9: Clear / Retrain buttons
		row_h = 0.06
		row_y -= (row_h + 0.02)
		self.ax_clear = self.fig.add_axes(panel_rect(pad_x, row_y, 0.48 - pad_x, row_h))
		self.btn_clear = Button(self.ax_clear, "Clear")
		self.btn_clear.on_clicked(self._on_clear)
		self.ax_retrain = self.fig.add_axes(panel_rect(0.52, row_y, 0.48 - pad_x, row_h))
		self.btn_retrain = Button(self.ax_retrain, "Retrain")
		self.btn_retrain.on_clicked(self._on_retrain)

		# Open a separate figure for the info table
		self._create_info_fig()
		self._update_info_fig()

		# connect click events
		self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

	# ---------- Callbacks ----------
	def _on_click(self, event):
		if event.inaxes != self.ax or event.button != 1:
			return
		x, y = event.xdata, event.ydata
		self.points.append((x, y))
		self.labels.append(self.selected_class)
		self._train_and_update()

	def _on_class_change(self, label):
		try:
			self.selected_class = self.class_names.index(label)
		except ValueError:
			self.selected_class = 0

	def _on_add_class(self, event):
		new_idx = len(self.class_names)
		self.class_names.append(f"Class {new_idx}")
		# Rebuild radio buttons with new labels
		self.ax_class.clear()
		self.radio_class = RadioButtons(self.ax_class, labels=[n for n in self.class_names], active=new_idx)
		self.radio_class.on_clicked(self._on_class_change)
		self.selected_class = new_idx
		self._train_and_update()
		self._update_info_fig()

	def _on_del_class(self, event):
		if len(self.class_names) <= 2:
			return
		del_idx = self.selected_class
		# remove points of this class
		keep = [i for i, y in enumerate(self.labels) if y != del_idx]
		self.points = [self.points[i] for i in keep]
		self.labels = [self.labels[i] for i in keep]
		# shift labels above del_idx down by 1
		self.labels = [y - 1 if y > del_idx else y for y in self.labels]
		# remove class name
		self.class_names.pop(del_idx)
		# rebuild radio
		self.ax_class.clear()
		self.radio_class = RadioButtons(self.ax_class, labels=[n for n in self.class_names], active=max(0, del_idx - 1))
		self.radio_class.on_clicked(self._on_class_change)
		self.selected_class = max(0, del_idx - 1)
		self._train_and_update()
		self._update_info_fig()

	def _on_act_change(self, label):
		self.activation = label
		self._train_and_update(reinit_model=True)
		self._update_info_fig()

	def _on_layers_change(self, text):
		try:
			# Extract integers robustly (labels are separate drawables now)
			tokens = [t.strip() for t in text.split(',') if t.strip()]
			vals = []
			for tok in tokens:
				m = re.search(r"-?\d+", tok)
				if m:
					vals.append(int(m.group(0)))
			if any(v <= 0 for v in vals):
				return
			self.hidden_layers = vals
			self._train_and_update(reinit_model=True)
			self._update_info_fig()
		except Exception:
			pass

	def _on_params_change(self, _):
		try:
			lr_text = self.tbx_lr.text
			m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", lr_text)
			if m:
				self.lr = float(m.group(0))

			epochs_text = self.tbx_epochs.text
			m = re.search(r"-?\d+", epochs_text)
			if m:
				self.epochs = int(m.group(0))

			batch_text = self.tbx_batch.text
			m = re.search(r"-?\d+", batch_text)
			if m:
				self.batch_size = int(m.group(0))

			l2_text = self.tbx_l2.text
			m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", l2_text)
			if m:
				self.l2 = float(m.group(0))
			self._train_and_update()
			self._update_info_fig()
		except Exception:
			pass
	def _enforce_prefix(self, tb: TextBox, prefix: str):
		# Normalize content to a single leading prefix and prevent duplication
		try:
			text = tb.text
			# Remove all known prefixes anywhere in the text
			clean = text
			for p in [self.layers_prefix, self.lr_prefix, self.epochs_prefix, self.batch_prefix, self.l2_prefix]:
				clean = clean.replace(p, "")
			clean = clean.strip()
			normalized = f"{prefix} {clean}".rstrip()
			if text != normalized:
				tb.set_val(normalized)
		except Exception:
			pass
			self._train_and_update()
			self._update_info_fig()
		except Exception:
			pass

	def _on_clear(self, event):
		self.points.clear()
		self.labels.clear()
		self._train_and_update()

	def _on_retrain(self, event):
		self._train_and_update(reinit_model=True)

	# ---------- Logic ----------
	def _train_and_update(self, reinit_model: bool = False):
		self._update_model(reinit_model=reinit_model)
		self._update_plot()

	def _update_model(self, reinit_model: bool = False):
		n_classes = max(len(self.class_names), max(self.labels) + 1 if self.labels else 0)
		if n_classes < 2:
			self.model = None
			return
		if self.model is None or reinit_model:
			self.model = MLP(input_dim=2, hidden_layers=self.hidden_layers, output_dim=n_classes,
							 activation=self.activation)
		# If output dim changed (class add/del), re-init
		if self.model.layers[-1] != n_classes:
			self.model = MLP(input_dim=2, hidden_layers=self.hidden_layers, output_dim=n_classes,
							 activation=self.activation)

		if len(self.points) >= n_classes:  # minimal data condition
			X = np.array(self.points, dtype=np.float64)
			y = np.array(self.labels, dtype=np.int64)
			self.model.fit(X, y, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size, l2=self.l2, verbose=False)

	def _update_plot(self):
		self.ax.clear()
		self.ax.set_xlim(-5, 5)
		self.ax.set_ylim(-5, 5)
		self.ax.grid(True, alpha=0.2)
		self.ax.set_title("Interactive MLP Decision Boundary")

		# draw decision boundary
		if self.model is not None:
			xx, yy, zz = self._compute_grid()
			if zz is not None:
				cmap = plt.get_cmap('tab10', len(self.class_names))
				self.ax.contourf(xx, yy, zz, levels=len(self.class_names), alpha=0.25, cmap=cmap)

		# scatter points per class
		plotted_any = False
		for c in range(len(self.class_names)):
			pts = [p for p, y in zip(self.points, self.labels) if y == c]
			if pts:
				plotted_any = True
				pts = np.array(pts)
				color = self.class_colors[c % len(self.class_colors)]
				self.ax.scatter(pts[:, 0], pts[:, 1], c=color, label=self.class_names[c], edgecolor='k', s=40)
		if plotted_any:
			self.ax.legend(loc='upper right')
		self.fig.canvas.draw_idle()

	def _create_info_fig(self):
		# Create a separate figure and table axes for info
		self.info_fig = plt.figure(figsize=(.0, 2.0))
		self.info_fig.canvas.manager.set_window_title("Model Bilgi Tablosu") if hasattr(self.info_fig.canvas.manager, 'set_window_title') else None
		self.ax_info_right = self.info_fig.add_axes([0.05, 0.05, 0.9, 0.9])
		self.ax_info_right.axis('off')
		col_labels = ["Parametre", "Değer", "Etki & Önerilen Aralık"]
		self.info_table_right = self.ax_info_right.table(cellText=[["", "", ""]]*7,
												 colLabels=col_labels,
												 colWidths=[0.21, 0.18, 0.66],
												 loc='center', cellLoc='left', colLoc='left')
		self.info_table_right.auto_set_font_size(False)
		self.info_table_right.set_fontsize(9)
		self.info_table_right.scale(1.05, 1.28)

	def _update_info_fig(self):
		if not hasattr(self, 'info_table_right'):
			return
		layers_str = ",".join(str(v) for v in self.hidden_layers) if self.hidden_layers else "-"
		n_layers = len(self.hidden_layers)
		if n_layers == 0:
			layers_note = "katman yok"
		elif n_layers == 1:
			layers_note = "1 katman"
		else:
			layers_note = f"{n_layers} katman"
		rows = [
			["Katmanlar (Layers)", f"{layers_str} → {layers_note}", "↑Derinlik = ↑Karmaşıklık, ↑Overfitting riski\nTipik: 2-4 katman, 4-256 nöron"],
			["Öğrenme Oranı (LR)", f"{self.lr}", "↑LR = ↑Hız, ↓Kararlılık\nTipik: 0.001-0.1"],
			["Epochs", f"{self.epochs}", "↑Epoch = ↑Doğruluk, ↑Overfitting\nTipik: 50-1000"],
			["Batch Size", f"{self.batch_size}", "↓Batch = ↑Hız, ↑Gürültü\nTipik: 16-128"],
			["L2 Regularization", f"{self.l2}", "↑L2 = ↑Basitlik, ↑Genelleme\nTipik: 0.0001-0.1"],
			["Sınıf Sayısı", f"{len(self.class_names)}", "Veriye bağlı (2-10)"],
			["Aktivasyon", self.activation, "ReLU: Hızlı\nTanh: Dengeli\nSigmoid: Çıkış katmanı"]
		]
		for r, row in enumerate(rows, start=1):
			for c, val in enumerate(row):
				self.info_table_right[(r, c)].get_text().set_text(str(val))
		headers = ["Parametre", "Değer", "Etki & Önerilen Aralık"]
		for c, h in enumerate(headers):
			self.info_table_right[(0, c)].get_text().set_text(h)
		self.ax_info_right.figure.canvas.draw_idle()

	# def _on_info_resize(self, event):
	# 	# Disabled resize handler - was causing table to disappear
	# 	pass

	def _compute_grid(self):
		try:
			gx = gy = 200
			x_min, x_max = -5, 5
			y_min, y_max = -5, 5
			xs = np.linspace(x_min, x_max, gx)
			ys = np.linspace(y_min, y_max, gy)
			xx, yy = np.meshgrid(xs, ys)
			grid = np.c_[xx.ravel(), yy.ravel()]
			if self.model is None:
				return xx, yy, None
			probs = self.model.predict_proba(grid)
			zz = np.argmax(probs, axis=1).reshape(xx.shape)
			return xx, yy, zz
		except Exception:
			return None, None, None


if __name__ == "__main__":
	app = InteractiveMLP()
	plt.show()
