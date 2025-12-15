import tkinter as tk
from tkinter import ttk, messagebox
import threading

# Keep imports exactly as they were
from rangeTree import range_tree_main
from kdtree import kdtree_main
from quadtree2 import movies_octree_main
from rTree import r_tree_main

class MovieSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multidimensional Structures GUI")
        self.root.geometry("1200x900")
        
        # --- THEME CONFIGURATION ---
        self.colors = {
            "bg": "#ffffff",
            "fg": "#1D2575",
            "accent": "#2c3e50",
            "highlight": "#2c3e50",
            "light_gray": "#f8f9fa"
        }
        
        self.root.configure(bg=self.colors["bg"])
        
        # Configure Styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # General Styles
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"], font=("Segoe UI", 10))
        self.style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.style.configure("TCheckbutton", background=self.colors["bg"], font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabelframe", background=self.colors["bg"], relief="flat")
        self.style.configure("TLabelframe.Label", background=self.colors["bg"], foreground=self.colors["accent"], font=("Segoe UI", 11, "bold"))
        
        # Input Styles
        self.style.configure("Card.TFrame", background=self.colors["light_gray"], relief="flat")
        
        # Header Style for Treeview
        self.style.configure("Treeview.Heading", 
                             background=self.colors["accent"], 
                             foreground="white", 
                             font=("Segoe UI", 10, "bold"),
                             relief="flat")
        self.style.map("Treeview.Heading", background=[('active', self.colors["accent"])])
        
        self.style.configure("Treeview", 
                             font=("Segoe UI", 10),
                             rowheight=30,
                             fieldbackground="white",
                             borderwidth=0)

        # --- MAIN CONTAINER ---
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)

        # --- HEADER / TREE SELECTION ---
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(header_frame, text="Select Tree Structure", font=("Segoe UI", 12, "bold"), foreground=self.colors["accent"]).pack(anchor=tk.CENTER)
        
        self.tree_method_var = tk.StringVar()
        self.tree_method_dropdown = ttk.Combobox(
            header_frame,
            textvariable=self.tree_method_var,
            state="readonly",
            width=30,
            values=["K-D Tree", "Quad Tree", "Range Tree", "R-Tree"],
            font=("Segoe UI", 10)
        )
        self.tree_method_dropdown.current(2)  # Default = Range Tree
        self.tree_method_dropdown.pack(anchor=tk.CENTER, pady=5)

        # --- FILTER SECTION (2x2 Grid) ---
        # We use a container with a subtle background to group filters
        filter_container = ttk.LabelFrame(main_container, text="  Search Filters  ", padding=15)
        filter_container.pack(fill=tk.X, pady=10)

        # Configure grid weights for symmetry
        filter_container.columnconfigure(0, weight=1)
        filter_container.columnconfigure(1, weight=1)

        # 1. Budget (Row 0, Col 0)
        self.create_filter_card(filter_container, "Budget", 0, 0)
        
        # 2. Popularity (Row 0, Col 1)
        self.create_filter_card(filter_container, "Popularity", 0, 1)
        
        # 3. Release Date (Row 1, Col 0)
        self.create_filter_card(filter_container, "Release Date", 1, 0)
        
        # 4. Genre (Row 1, Col 1)
        self.create_filter_card(filter_container, "Genre", 1, 1)

        # --- INFO TEXT & SEARCH BUTTON ---
        action_frame = ttk.Frame(main_container)
        action_frame.pack(fill=tk.X, pady=15)

        info_text = ttk.Label(action_frame, 
                             text="Enter Min/Max for numeric values. Dates: YYYY-MM-DD. Separate genres with spaces.",
                             foreground="#7f8c8d", font=("Segoe UI", 9, "italic"))
        info_text.pack(side=tk.TOP, pady=(0, 10))

        self.search_btn = tk.Button(action_frame, text="SEARCH MOVIES", command=self.search_movies,
                                   bg=self.colors["highlight"], fg='white',
                                   font=('Segoe UI', 11, 'bold'),
                                   relief="flat", cursor='hand2', 
                                   padx=40, pady=10)
        self.search_btn.pack(side=tk.TOP)

        # --- RESULTS SECTION ---
        results_frame = ttk.Frame(main_container)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        results_lbl = ttk.Label(results_frame, text="Results", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"])
        results_lbl.pack(anchor=tk.W, pady=(0,5))

        # Create Treeview for table
        columns = ('Origin Country', 'Budget', 'Popularity', 'Release Date', 'Genre', 'Revenue')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='tree headings', height=10)

        columns_setup = [
            ('#0', 'Title', tk.W, 250),
            ('Origin Country', 'Country', tk.W, 100),
            ('Budget', 'Budget', tk.CENTER, 100),
            ('Popularity', 'Popularity', tk.CENTER, 80),
            ('Release Date', 'Date', tk.CENTER, 100),
            ('Genre', 'Genre', tk.W, 250),
            ('Revenue', 'Revenue', tk.CENTER, 100)
        ]

        for col_id, col_name, anchor_pos, width in columns_setup:
            self.tree.heading(
                col_id,
                text=col_name,
                anchor=anchor_pos,
                command=lambda c=col_id: self.sort_treeview_column(c, False)
            )
            self.tree.column(col_id, width=width, anchor=anchor_pos)

        # Scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        # Layout Treeview
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Status bar
        self.status_label = tk.Label(root, text="Ready", bd=0, bg=self.colors["accent"], fg="white",
                                    relief=tk.FLAT, anchor=tk.W, font=('Segoe UI', 9), padx=10, pady=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_filter_card(self, parent, title, row, col):
        """Helper to create symmetric filter cards"""
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
        
        # Header within card
        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(fill=tk.X, anchor=tk.W)

        # Logic Mapping based on title
        if title == "Budget":
            self.budget_var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(header, text=title.upper(), variable=self.budget_var, command=self.toggle_budget, style="TCheckbutton")
            cb.pack(side=tk.LEFT)
            
            self.budget_frame = ttk.Frame(card, style="Card.TFrame")
            self.budget_from_label = ttk.Label(self.budget_frame, text="Min", style="TLabel")
            self.budget_min = ttk.Entry(self.budget_frame, width=12)
            self.budget_to_label = ttk.Label(self.budget_frame, text="Max", style="TLabel")
            self.budget_max = ttk.Entry(self.budget_frame, width=12)

        elif title == "Popularity":
            self.popularity_var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(header, text=title.upper(), variable=self.popularity_var, command=self.toggle_popularity, style="TCheckbutton")
            cb.pack(side=tk.LEFT)

            self.popularity_frame = ttk.Frame(card, style="Card.TFrame")
            self.popularity_from_label = ttk.Label(self.popularity_frame, text="Min", style="TLabel")
            self.popularity_min = ttk.Entry(self.popularity_frame, width=12)
            self.popularity_to_label = ttk.Label(self.popularity_frame, text="Max", style="TLabel")
            self.popularity_max = ttk.Entry(self.popularity_frame, width=12)

        elif title == "Release Date":
            self.date_var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(header, text=title.upper(), variable=self.date_var, command=self.toggle_date, style="TCheckbutton")
            cb.pack(side=tk.LEFT)

            self.date_frame = ttk.Frame(card, style="Card.TFrame")
            self.date_from_label = ttk.Label(self.date_frame, text="Start", style="TLabel")
            self.date_min = ttk.Entry(self.date_frame, width=12)
            self.date_to_label = ttk.Label(self.date_frame, text="End", style="TLabel")
            self.date_max = ttk.Entry(self.date_frame, width=12)

        elif title == "Genre":
            self.genre_var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(header, text=title.upper(), variable=self.genre_var, command=self.toggle_genre, style="TCheckbutton")
            cb.pack(side=tk.LEFT)

            self.genre_frame = ttk.Frame(card, style="Card.TFrame")
            self.genre_keywords_label = ttk.Label(self.genre_frame, text="Key:", style="TLabel")
            self.genre_keywords = ttk.Entry(self.genre_frame, width=15)
            self.neighbors_label = ttk.Label(self.genre_frame, text="Num:", style="TLabel")
            self.number_of_results_entry = ttk.Entry(self.genre_frame, width=5)

    def toggle_budget(self):
        if self.budget_var.get():
            self.budget_frame.pack(fill=tk.X, pady=(10, 0))
            self.budget_from_label.pack(side=tk.LEFT, padx=(0,5))
            self.budget_min.pack(side=tk.LEFT, padx=(0,10))
            self.budget_to_label.pack(side=tk.LEFT, padx=(0,5))
            self.budget_max.pack(side=tk.LEFT)
        else:
            self.budget_frame.pack_forget()

    def toggle_popularity(self):
        if self.popularity_var.get():
            self.popularity_frame.pack(fill=tk.X, pady=(10, 0))
            self.popularity_from_label.pack(side=tk.LEFT, padx=(0,5))
            self.popularity_min.pack(side=tk.LEFT, padx=(0,10))
            self.popularity_to_label.pack(side=tk.LEFT, padx=(0,5))
            self.popularity_max.pack(side=tk.LEFT)
        else:
            self.popularity_frame.pack_forget()

    def toggle_date(self):
        if self.date_var.get():
            self.date_frame.pack(fill=tk.X, pady=(10, 0))
            self.date_from_label.pack(side=tk.LEFT, padx=(0,5))
            self.date_min.pack(side=tk.LEFT, padx=(0,10))
            self.date_to_label.pack(side=tk.LEFT, padx=(0,5))
            self.date_max.pack(side=tk.LEFT)
        else:
            self.date_frame.pack_forget()

    def toggle_genre(self):
        if self.genre_var.get():
            self.genre_frame.pack(fill=tk.X, pady=(10, 0))
            self.genre_keywords_label.pack(side=tk.LEFT, padx=(0,5))
            self.genre_keywords.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
            self.neighbors_label.pack(side=tk.LEFT, padx=(0,5))
            self.number_of_results_entry.pack(side=tk.LEFT)
        else:
            self.genre_frame.pack_forget()

    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=f"  {message}")
        self.root.update_idletasks()

    def search_movies(self):
        """Execute movie search in a separate thread"""
        self.search_btn.config(state='disabled', text="SEARCHING...")
        self.update_status("Executing search algorithm...")

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Run search in separate thread
        thread = threading.Thread(target=self.run_search)
        thread.daemon = True
        thread.start()

    def run_search(self):
        """Run the actual search"""

        conditions = {}
        # Budget condition
        if self.budget_var.get():
            budget_min = float(self.budget_min.get()) if self.budget_min.get() else 0
            budget_max = float(self.budget_max.get()) if self.budget_max.get() else 999999999
            conditions['budget'] = (budget_min, budget_max)
        else:
            conditions['budget'] = (0, 999999999)

        # Popularity condition
        if self.popularity_var.get():
            pop_min = float(self.popularity_min.get()) if self.popularity_min.get() else 0
            pop_max = float(self.popularity_max.get()) if self.popularity_max.get() else 999999
            conditions['popularity'] = (pop_min, pop_max)
        else:
            conditions['popularity'] = (0, 999999)

        # Date condition
        if self.date_var.get():
            date_min = self.date_min.get() if self.date_min.get() else '1900-01-01'
            date_max = self.date_max.get() if self.date_max.get() else '2030-12-31'
            conditions['release_date'] = (date_min, date_max)
        else:
            conditions['release_date'] = ('1900-01-01', '2030-12-31')

        # Genre keywords
        genre_kw = None
        num_of_results = None
        if self.genre_var.get():
            genre_kw_raw = self.genre_keywords.get().strip()
            if genre_kw_raw:
                genre_kw = genre_kw_raw.replace(",", " ")
                num_of_results = int(self.number_of_results_entry.get()) if self.number_of_results_entry.get() else None

        # Call search function
        tree_choice = self.tree_method_var.get()

        try:
            if tree_choice == "Range Tree":
                results = range_tree_main(conditions, genre_kw, num_of_results)
            elif tree_choice == "Quad Tree":
                results = movies_octree_main(conditions, genre_kw, num_of_results)
            elif tree_choice == "K-D Tree":
                results = kdtree_main(conditions, genre_kw, num_of_results)
            elif tree_choice == "R-Tree":
                results = r_tree_main(conditions, genre_kw, num_of_results)
            else:
                raise ValueError("No tree structure selected.")
            
            # Display results in GUI thread
            self.root.after(0, lambda: self.display_results(results))
        except Exception as e:
             self.root.after(0, lambda: self.show_error(str(e)))

    def display_results(self, results):
        """Display search results in the table"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not results:
            self.update_status("No results found")
            messagebox.showinfo("No Results", "No movies found matching your criteria.")
        else:
            for i, row in enumerate(results):
                # Data extraction
                title = str(row[1]) if len(row) > 1 else "N/A"
                origin = str(row[3]).replace("['", "").replace("']", "") if len(row) > 3 else "N/A"
                budget = f"${float(row[8]):,.0f}" if len(row) > 8 else "$0"
                popularity = f"{float(row[11]):.2f}" if len(row) > 11 else "N/A"
                date = str(row[5]) if len(row) > 5 else "N/A"
                genre = str(row[6]) if len(row) > 6 else "N/A"
                revenue = f"${float(row[9]):,.0f}" if len(row) > 9 else "$0"

                # Tag logic for alternating colors
                tag = 'even' if i % 2 == 0 else 'odd'
                
                self.tree.insert('', tk.END, text=title,
                               values=(origin, budget, popularity, date, genre, revenue), tags=(tag,))

            self.tree.tag_configure('odd', background=self.colors["light_gray"])
            self.tree.tag_configure('even', background="white")
            self.update_status(f"Found {len(results)} movies")

        # Re-enable button
        self.search_btn.config(state='normal', text="SEARCH MOVIES")

    def show_error(self, error_msg):
        """Show error message"""
        self.update_status("Error during search")
        self.search_btn.config(state='normal', text="SEARCH MOVIES")
        messagebox.showerror("Error", f"An error occurred:\n{error_msg}")

    def sort_treeview_column(self, col, reverse):
        """Sort treeview content when a column header is clicked."""
        l = []
        for k in self.tree.get_children(''):
            if col == '#0':
                val = self.tree.item(k, 'text')
            else:
                val = self.tree.set(k, col)
            l.append((val, k))

        def convert_value(value):
            clean_val = str(value).replace('$', '').replace(',', '').strip()
            try:
                return float(clean_val)
            except ValueError:
                return clean_val.lower()

        l.sort(key=lambda t: convert_value(t[0]), reverse=reverse)

        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)
            # Re-apply striping
            tag = 'even' if index % 2 == 0 else 'odd'
            self.tree.item(k, tags=(tag,))

        self.tree.heading(col, command=lambda: self.sort_treeview_column(col, not reverse))


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieSearchGUI(root)
    root.mainloop()