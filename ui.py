from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk


class MainWindow:
    def __init__(self, root: tk.Tk, app):
        self.root = root
        self.app = app

        self.root.title("Sports Toolkit")
        self.root.geometry("1500x980")
        self.root.configure(bg="#1f1f1f")

        self.current_pil = None
        self.current_tk = None
        self.current_scale = 1.0
        self.overlay_boxes = []
        self.manual_boxes = []
        self.manual_selected_ids = set()
        self.manual_hitboxes = []

        self.debug_views = []
        self.debug_tk_images = []

        self.module_title_var = tk.StringVar(value="AI Crop Tool")
        self.module_subtitle_var = tk.StringVar(value="Automated subject-driven crop recommendations")
        self.profile_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self.input_folder_var = tk.StringVar(value="Input: —")
        self.output_folder_var = tk.StringVar(value="Output: —")
        self.filename_var = tk.StringVar(value="File: No image loaded")
        self.index_var = tk.StringVar(value="0 / 0")

        self._build()

    def _build(self):
        self.topbar = tk.Frame(self.root, bg="#252525", height=84)
        self.topbar.pack(side=tk.TOP, fill=tk.X)
        self.topbar.pack_propagate(False)

        self.main_area = tk.Frame(self.root, bg="#1f1f1f")
        self.main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left = tk.Frame(self.main_area, bg="#2a2a2a", width=340)
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.left.pack_propagate(False)

        self.center = tk.Frame(self.main_area, bg="#1f1f1f")
        self.center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_topbar()
        self._build_left()
        self._build_center()

        self.root.bind("<Return>", lambda e: self.app.approve())
        self.root.bind("<Right>", lambda e: self.app.next_image())
        self.root.bind("<Left>", lambda e: self.app.previous_image())

    def _build_topbar(self):
        row1 = tk.Frame(self.topbar, bg="#252525")
        row1.pack(fill=tk.X, padx=8, pady=(6, 2))

        row2 = tk.Frame(self.topbar, bg="#252525")
        row2.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(row1, text="Module", bg="#252525", fg="#dddddd", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(6, 4))

        tool_names = self.app.get_tool_display_names()
        initial_tool_name = tool_names[0] if tool_names else "AI Crop Tool"
        self.tool_var = tk.StringVar(value=initial_tool_name)

        self.tool_menu = tk.OptionMenu(row1, self.tool_var, initial_tool_name)
        tool_menu = self.tool_menu["menu"]
        tool_menu.delete(0, "end")
        for name in tool_names:
            tool_menu.add_command(label=name, command=tk._setit(self.tool_var, name, self._on_tool_selected))

        self.tool_menu.config(
            bg="#3a3a3a",
            fg="white",
            highlightthickness=0,
            activebackground="#4a4a4a",
            activeforeground="white",
            width=18,
        )
        self.tool_menu.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(row1, text="Sport/Profile", bg="#252525", fg="#dddddd", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(8, 4))

        profile_names = self.app.get_profile_names()
        initial_profile = profile_names[0] if profile_names else "Generic Sport"
        self.profile_var.set(initial_profile)

        self.profile_menu = tk.OptionMenu(row1, self.profile_var, initial_profile)
        profile_menu = self.profile_menu["menu"]
        profile_menu.delete(0, "end")
        for name in profile_names:
            profile_menu.add_command(label=name, command=tk._setit(self.profile_var, name, self._on_profile_selected))

        self.profile_menu.config(
            bg="#3a3a3a",
            fg="white",
            highlightthickness=0,
            activebackground="#4a4a4a",
            activeforeground="white",
            width=14,
        )
        self.profile_menu.pack(side=tk.LEFT, padx=(0, 4))

        tk.Button(row1, text="Save Sport", command=self._save_current_profile, bg="#6c63ff", fg="white").pack(side=tk.LEFT, padx=(4, 4))
        tk.Button(row1, text="Save As New", command=self._save_as_new_profile, bg="#7b8a8b", fg="white").pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(row1, text="Input Folder", command=self._choose_input, bg="#4a4a4a", fg="white").pack(side=tk.LEFT, padx=(8, 4))
        tk.Button(row1, text="Output Folder", command=self._choose_output, bg="#4a4a4a", fg="white").pack(side=tk.LEFT, padx=4)

        tk.Button(row1, text="Start Batch", command=self.app.start_batch, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(row1, text="Approve", command=self.app.approve, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=4)
        tk.Button(row1, text="Prev", command=self.app.previous_image, bg="#555555", fg="white").pack(side=tk.LEFT, padx=(12, 4))
        tk.Button(row1, text="Next", command=self.app.next_image, bg="#555555", fg="white").pack(side=tk.LEFT, padx=4)

        tk.Label(row2, textvariable=self.input_folder_var, bg="#252525", fg="#d8d8d8", font=("Arial", 9)).pack(side=tk.LEFT, padx=(6, 14))
        tk.Label(row2, textvariable=self.output_folder_var, bg="#252525", fg="#d8d8d8", font=("Arial", 9)).pack(side=tk.LEFT, padx=(0, 14))
        tk.Label(row2, textvariable=self.filename_var, bg="#252525", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 14))
        tk.Label(row2, textvariable=self.index_var, bg="#252525", fg="#9ad0ff", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 14))

        status_frame = tk.Frame(row2, bg="#252525")
        status_frame.pack(side=tk.RIGHT)

        tk.Label(status_frame, text="Status:", bg="#252525", fg="#bbbbbb", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(status_frame, textvariable=self.status_var, bg="#252525", fg="white", font=("Arial", 9)).pack(side=tk.LEFT)

    def _build_left(self):
        self.left_canvas = tk.Canvas(self.left, bg="#2a2a2a", highlightthickness=0)
        self.left_scrollbar = tk.Scrollbar(self.left, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        self.left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.left_inner = tk.Frame(self.left_canvas, bg="#2a2a2a")
        self.left_window = self.left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")

        self.left_inner.bind("<Configure>", self._on_left_inner_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        banner = tk.Frame(self.left_inner, bg="#1d3557", bd=0, highlightthickness=0)
        banner.pack(fill="x", padx=10, pady=(10, 12))
        self.module_banner = banner

        tk.Label(
            banner,
            textvariable=self.module_title_var,
            bg="#1d3557",
            fg="white",
            font=("Arial", 14, "bold"),
            anchor="w",
            justify=tk.LEFT,
        ).pack(fill="x", padx=12, pady=(10, 2))
        self.module_title_label = banner.winfo_children()[-1]

        tk.Label(
            banner,
            textvariable=self.module_subtitle_var,
            bg="#1d3557",
            fg="#dbe7f5",
            font=("Arial", 9),
            anchor="w",
            justify=tk.LEFT,
            wraplength=280,
        ).pack(fill="x", padx=12, pady=(0, 10))
        self.module_subtitle_label = banner.winfo_children()[-1]

        tk.Label(self.left_inner, text="Module Controls", bg="#2a2a2a", fg="white", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 6))

        self.dynamic_panel = tk.Frame(self.left_inner, bg="#2a2a2a")
        self.dynamic_panel.pack(fill="both", expand=True, padx=0, pady=(0, 12))

    def _build_center(self):
        self.canvas = tk.Canvas(self.center, bg="black", highlightthickness=0, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        thumb_frame = tk.Frame(self.center, bg="#2b2b2b", height=120)
        thumb_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 6))
        thumb_frame.pack_propagate(False)

        self.thumb_list = tk.Listbox(thumb_frame, height=5, bg="#111111", fg="white", selectbackground="#2196F3")
        self.thumb_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.thumb_list.bind("<<ListboxSelect>>", self._on_thumbnail_selected)

        log_frame = tk.Frame(self.center, bg="#191919", height=120)
        log_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        log_frame.pack_propagate(False)

        tk.Label(log_frame, text="Log", bg="#191919", fg="white", font=("Arial", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 2))

        self.log_console = ScrolledText(log_frame, height=5, bg="black", fg="lime", font=("Consolas", 8))
        self.log_console.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_console.config(state="disabled")

    def _on_left_inner_configure(self, event=None):
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event):
        self.left_canvas.itemconfigure(self.left_window, width=event.width)

    def _on_mousewheel(self, event):
        widget_under_mouse = self.root.winfo_containing(event.x_root, event.y_root)
        if widget_under_mouse is None:
            return

        parent = widget_under_mouse
        while parent is not None:
            if parent == self.left_canvas or parent == self.left_inner:
                self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                return
            parent_name = parent.winfo_parent()
            if not parent_name:
                break
            try:
                parent = self.root.nametowidget(parent_name)
            except Exception:
                break

    def _choose_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.app.set_input_folder(folder)

    def _choose_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.app.set_output_folder(folder)

    def _save_current_profile(self):
        self.app.save_current_profile_from_active_tool()

    def _save_as_new_profile(self):
        new_name = simpledialog.askstring("Save As New Sport", "Enter new sport/profile name:", parent=self.root)
        if new_name:
            self.app.save_current_profile_from_active_tool(new_name=new_name)

    def _on_tool_selected(self, display_name):
        self.app.set_active_tool_by_display_name(display_name)

    def _on_profile_selected(self, profile_name):
        self.app.on_profile_changed(profile_name)

    def _on_thumbnail_selected(self, event=None):
        sel = self.thumb_list.curselection()
        if sel:
            self.app.select_image_index(sel[0])

    def _on_canvas_click(self, event):
        for item in self.manual_hitboxes:
            x1, y1, x2, y2 = item["bbox"]
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.app.on_manual_detection_clicked(item["id"])
                return

    def set_tool_panel(self, panel_builder):
        for child in self.dynamic_panel.winfo_children():
            child.destroy()
        panel = panel_builder(self.dynamic_panel)
        panel.pack(fill="both", expand=True)
        self.left_canvas.update_idletasks()
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        self.left_canvas.yview_moveto(0)

    def set_active_tool_label(self, display_name: str):
        self.tool_var.set(display_name)
        self.module_title_var.set(display_name)

        subtitles = {
            "AI Crop Tool": "Automated subject-driven crop recommendations",
            "Manual Crop Tool": "Manual subject selection and crop building",
            "AI Cull Tool": "AI-assisted image culling, ranking, and filtering",
            "Cached Crop Tool": "Cache-driven crop preview with nested ratio guides and batch commit",
            "Full Trial Pipeline": "Automated cull-to-crop batch run with live visual progress",
            "LM Studio Test": "LM Studio local model test and vision prompt debugging",
        }

        banner_colors = {
            "AI Crop Tool": ("#1d3557", "#dbe7f5"),
            "Manual Crop Tool": ("#6a3f00", "#ffe7c2"),
            "AI Cull Tool": ("#3f2a56", "#eadbff"),
            "Cached Crop Tool": ("#5a3a1a", "#ffe8cf"),
            "Full Trial Pipeline": ("#1f5c42", "#d8ffef"),
            "LM Studio Test": ("#3f2a56", "#eadbff"),
        }

        subtitle = subtitles.get(display_name, "")
        banner_bg, subtitle_fg = banner_colors.get(display_name, ("#1d3557", "#dbe7f5"))

        self.module_subtitle_var.set(subtitle)
        self.module_banner.configure(bg=banner_bg)
        self.module_title_label.configure(bg=banner_bg)
        self.module_subtitle_label.configure(bg=banner_bg, fg=subtitle_fg)
        self.status_var.set(f"In {display_name}")

    def set_profiles(self, names: list[str]):
        menu = self.profile_menu["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=tk._setit(self.profile_var, name, self._on_profile_selected))
        if names:
            current = self.app.get_selected_profile_name()
            self.profile_var.set(current if current in names else names[0])

    def set_selected_profile(self, profile_name: str):
        self.profile_var.set(profile_name)

    def set_header_info(self, input_name: str, output_name: str, filename: str, index_text: str):
        self.input_folder_var.set(f"Input: {input_name}")
        self.output_folder_var.set(f"Output: {output_name}")
        self.filename_var.set(f"File: {filename}")
        self.index_var.set(index_text)

    def append_log(self, message: str):
        self.log_console.config(state="normal")
        self.log_console.insert("end", message + "\n")
        self.log_console.see("end")
        self.log_console.config(state="disabled")
        self.status_var.set(message)

    def set_thumbnail_paths(self, paths: list[Path]):
        self.thumb_list.delete(0, "end")
        for path in paths:
            self.thumb_list.insert("end", path.name)

    def highlight_thumbnail_index(self, index: int):
        self.thumb_list.selection_clear(0, "end")
        if index >= 0:
            self.thumb_list.selection_set(index)
            self.thumb_list.see(index)

    def clear_image(self):
        self.current_pil = None
        self.current_tk = None
        self.overlay_boxes = []
        self.manual_boxes = []
        self.manual_selected_ids = set()
        self.manual_hitboxes = []
        self.debug_views = []
        self.debug_tk_images = []
        self.canvas.delete("all")

    def show_image(self, pil_image):
        self.current_pil = pil_image
        self.redraw_canvas()

    def set_overlay_boxes(self, boxes):
        self.overlay_boxes = boxes
        if not self.debug_views:
            self.redraw_canvas()

    def set_manual_boxes(self, boxes):
        self.manual_boxes = boxes
        if not self.debug_views:
            self.redraw_canvas()

    def set_manual_selected_ids(self, selected_ids):
        self.manual_selected_ids = set(selected_ids)
        if not self.debug_views:
            self.redraw_canvas()

    def set_debug_views(self, views):
        self.debug_views = list(views or [])
        self.redraw_canvas()

    def clear_debug_views(self):
        self.debug_views = []
        self.debug_tk_images = []
        self.redraw_canvas()

    def _load_debug_image(self, value):
        if value is None:
            return None
        if hasattr(value, "copy") and hasattr(value, "size"):
            return value
        try:
            path = Path(value)
            if path.exists():
                return Image.open(path).convert("RGB")
        except Exception:
            return None
        return None

    def _draw_debug_grid(self):
        self.canvas.delete("all")
        self.manual_hitboxes = []
        self.debug_tk_images = []

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        pad = 10
        header_h = 26
        cell_w = max(100, (cw - pad * 3) // 2)
        cell_h = max(100, (ch - pad * 3) // 2)

        panels = list(self.debug_views[:4])
        while len(panels) < 4:
            panels.append(("Empty", None))

        for idx, (title, content) in enumerate(panels):
            row = idx // 2
            col = idx % 2

            x = pad + col * (cell_w + pad)
            y = pad + row * (cell_h + pad)

            self.canvas.create_rectangle(x, y, x + cell_w, y + cell_h, outline="#555555", width=1)
            self.canvas.create_rectangle(x, y, x + cell_w, y + header_h, fill="#202020", outline="#555555", width=1)
            self.canvas.create_text(x + 8, y + header_h / 2, text=title, fill="white", anchor="w", font=("Arial", 10, "bold"))

            pil = self._load_debug_image(content)
            if pil is None:
                self.canvas.create_text(
                    x + cell_w / 2,
                    y + cell_h / 2,
                    text="No preview",
                    fill="#AAAAAA",
                    font=("Arial", 12),
                )
                continue

            avail_w = cell_w - 12
            avail_h = cell_h - header_h - 12
            iw, ih = pil.size
            scale = min(avail_w / iw, avail_h / ih)
            new_w = max(1, int(iw * scale))
            new_h = max(1, int(ih * scale))

            resized = pil.resize((new_w, new_h))
            tk_img = ImageTk.PhotoImage(resized)
            self.debug_tk_images.append(tk_img)

            img_x = x + (cell_w - new_w) // 2
            img_y = y + header_h + (avail_h - new_h) // 2
            self.canvas.create_image(img_x, img_y, image=tk_img, anchor="nw")

    def redraw_canvas(self):
        if self.debug_views:
            self._draw_debug_grid()
            return

        self.canvas.delete("all")
        self.manual_hitboxes = []

        if self.current_pil is None:
            return

        self.canvas.update_idletasks()
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        iw, ih = self.current_pil.size

        scale = min(cw / iw, ch / ih)
        self.current_scale = scale
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))

        resized = self.current_pil.resize((new_w, new_h))
        self.current_tk = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, image=self.current_tk, anchor="nw")

        for det in self.manual_boxes:
            x1 = det.bbox.x1 * scale
            y1 = det.bbox.y1 * scale
            x2 = det.bbox.x2 * scale
            y2 = det.bbox.y2 * scale
            color = "#FFFFFF" if det.id in self.manual_selected_ids else det.color
            width = 3 if det.id in self.manual_selected_ids else 2

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=(6, 4))
            txt = self.canvas.create_text(x1 + 8, y1 + 8, text=str(det.id), fill=color, anchor="nw", font=("Arial", 13, "bold"))
            bbox = self.canvas.bbox(txt)
            if bbox:
                bg = self.canvas.create_rectangle(bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2, fill="black", outline=color, width=1)
                self.canvas.tag_lower(bg, txt)
                self.manual_hitboxes.append(
                    {
                        "id": det.id,
                        "bbox": (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4),
                    }
                )

            info = self.canvas.create_text(x1 + 8, y1 + 28, text=det.label, fill=det.color, anchor="nw", font=("Arial", 9, "bold"))
            info_bbox = self.canvas.bbox(info)
            if info_bbox:
                bg2 = self.canvas.create_rectangle(info_bbox[0] - 2, info_bbox[1] - 2, info_bbox[2] + 2, info_bbox[3] + 2, fill="black", outline="")
                self.canvas.tag_lower(bg2, info)

        for crop in self.overlay_boxes:
            x1 = crop.bbox.x1 * scale
            y1 = crop.bbox.y1 * scale
            x2 = crop.bbox.x2 * scale
            y2 = crop.bbox.y2 * scale
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=crop.color, width=3)
            if crop.name:
                text_id = self.canvas.create_text(x1 + 6, y1 + 20, text=crop.name, fill=crop.color, anchor="nw", font=("Arial", 10, "bold"))
                tb = self.canvas.bbox(text_id)
                if tb:
                    bg = self.canvas.create_rectangle(tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2, fill="black", outline="")
                    self.canvas.tag_lower(bg, text_id)