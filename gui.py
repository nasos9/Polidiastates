import tkinter as tk
from tkinter import ttk, messagebox
import threading

from rangeTree import range_tree_main
from kdtree import kdtree_main
from quadtree2 import movies_octree_main
from rTree import r_tree_main

class MovieSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multidimensional structures GUI")
        self.root.geometry("1200x750")
        self.root.configure(bg='#f5f5f5')

        # Main container
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tree selection
        tree_method_container = ttk.Frame(main_container)
        tree_method_container.pack(fill=tk.X, pady=10)

        ttk.Label(tree_method_container, text="Select Tree Structure:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT,
                                                                                                         padx=5)

        self.tree_method_var = tk.StringVar()
        self.tree_method_dropdown = ttk.Combobox(
            tree_method_container,
            textvariable=self.tree_method_var,
            state="readonly",
            width=20,
            values=["K-D Tree", "Quad Tree", "Range Tree", "R-Tree"]
        )
        self.tree_method_dropdown.current(2)  # Default = Range Tree
        self.tree_method_dropdown.pack(side=tk.LEFT, padx=10)

        # Top section: Input form
        form_frame = ttk.LabelFrame(main_container, text="Select attributes (0-4)", padding=15)
        form_frame.pack(fill=tk.X, padx=5, pady=5)
        # Top section: Input form
        form_frame = ttk.LabelFrame(main_container, text="Select attributes (0-4)", padding=15)
        form_frame.pack(fill=tk.X, padx=5, pady=5)

        # ===== CHECKBOXES AND INPUT FIELDS =====

        # Budget
        budget_container = ttk.Frame(form_frame)
        budget_container.pack(fill=tk.X, pady=5)

        self.budget_var = tk.BooleanVar(value=False)
        budget_check = ttk.Checkbutton(budget_container, text="budget", variable=self.budget_var,
                                       command=self.toggle_budget)
        budget_check.pack(side=tk.LEFT, padx=5)

        self.budget_frame = ttk.Frame(budget_container)
        self.budget_from_label = ttk.Label(self.budget_frame, text="από")
        self.budget_min = ttk.Entry(self.budget_frame, width=15)
        self.budget_to_label = ttk.Label(self.budget_frame, text="έως")
        self.budget_max = ttk.Entry(self.budget_frame, width=15)

        # Popularity
        popularity_container = ttk.Frame(form_frame)
        popularity_container.pack(fill=tk.X, pady=5)

        self.popularity_var = tk.BooleanVar(value=False)
        popularity_check = ttk.Checkbutton(popularity_container, text="popularity",
                                          variable=self.popularity_var,
                                          command=self.toggle_popularity)
        popularity_check.pack(side=tk.LEFT, padx=5)

        self.popularity_frame = ttk.Frame(popularity_container)
        self.popularity_from_label = ttk.Label(self.popularity_frame, text="από")
        self.popularity_min = ttk.Entry(self.popularity_frame, width=15)
        self.popularity_to_label = ttk.Label(self.popularity_frame, text="έως")
        self.popularity_max = ttk.Entry(self.popularity_frame, width=15)

        # Release Date
        date_container = ttk.Frame(form_frame)
        date_container.pack(fill=tk.X, pady=5)

        self.date_var = tk.BooleanVar(value=False)
        date_check = ttk.Checkbutton(date_container, text="release_date",
                                     variable=self.date_var,
                                     command=self.toggle_date)
        date_check.pack(side=tk.LEFT, padx=5)

        self.date_frame = ttk.Frame(date_container)
        self.date_from_label = ttk.Label(self.date_frame, text="από")
        self.date_min = ttk.Entry(self.date_frame, width=15)
        self.date_to_label = ttk.Label(self.date_frame, text="έως")
        self.date_max = ttk.Entry(self.date_frame, width=15)

        # Genre Keywords
        genre_container = ttk.Frame(form_frame)
        genre_container.pack(fill=tk.X, pady=5)

        self.genre_var = tk.BooleanVar(value=False)
        genre_check = ttk.Checkbutton(genre_container, text="genre",
                                      variable=self.genre_var,
                                      command=self.toggle_genre)
        genre_check.pack(side=tk.LEFT, padx=5)

        self.genre_frame = ttk.Frame(genre_container)
        self.genre_keywords_label = ttk.Label(self.genre_frame, text="Keywords:")
        self.genre_keywords = ttk.Entry(self.genre_frame, width=25)
        self.neighbors_label = ttk.Label(self.genre_frame, text="Number of Results:")
        self.number_of_results_entry = ttk.Entry(self.genre_frame, width=10)

        # Info text
        info_text = ttk.Label(form_frame,
                             text="Enter conditions: For the numeric attributes: enter min value in the left box and max value in the right box. For genres you can either seperate with commas (,) or with just with space. Dates are in YYYY-MM-DD format (e.g., 2005-12-31). Leave boxes empty for no limit.",
                             wraplength=900, justify=tk.LEFT, foreground='#666')
        info_text.pack(pady=10)

        # Search Button
        self.search_btn = tk.Button(form_frame, text="Search", command=self.search_movies,
                                    bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                                    cursor='hand2', padx=30, pady=8)
        self.search_btn.pack(pady=15)

        # ===== RESULTS SECTION =====
        results_frame = ttk.LabelFrame(main_container,
                                      text="Search results: You can tap on the headings to sort the table based on the attribute you want",
                                      padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create Treeview for table
        columns = ('Origin Country', 'Budget', 'Popularity', 'Release Date', 'Genre', 'Revenue')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='tree headings', height=15)

        columns_setup = [
            ('#0', 'Title', tk.W),
            ('Origin Country', 'Origin Country', tk.W),
            ('Budget', 'Budget', tk.CENTER),
            ('Popularity', 'Popularity', tk.CENTER),
            ('Release Date', 'Release Date', tk.CENTER),
            ('Genre', 'Genre', tk.W),
            ('Revenue', 'Revenue', tk.CENTER)
        ]

        for col_id, col_name, anchor_pos in columns_setup:
            self.tree.heading(
                col_id,
                text=col_name,
                anchor=anchor_pos,
                # Το lambda c=col_id είναι απαραίτητο για να θυμάται τη σωστή στήλη
                command=lambda c=col_id: self.sort_treeview_column(c, False)
            )

        # Define column widths
        self.tree.column('#0', width=250, anchor=tk.W)
        self.tree.column('Origin Country', width=120, anchor=tk.W)
        self.tree.column('Budget', width=120, anchor=tk.CENTER)
        self.tree.column('Popularity', width=100, anchor=tk.CENTER)
        self.tree.column('Release Date', width=100, anchor=tk.CENTER)
        self.tree.column('Genre', width=280, anchor=tk.W)
        self.tree.column('Revenue', width=120, anchor=tk.CENTER)

        # Add scrollbars
        vsb = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        # Status bar
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN,
                                    anchor=tk.W, font=('Arial', 9))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_budget(self):
        """Show/hide budget input fields"""
        if self.budget_var.get():
            self.budget_frame.pack(side=tk.LEFT, padx=10)
            self.budget_from_label.pack(side=tk.LEFT, padx=5)
            self.budget_min.pack(side=tk.LEFT, padx=5)
            self.budget_to_label.pack(side=tk.LEFT, padx=5)
            self.budget_max.pack(side=tk.LEFT, padx=5)
        else:
            self.budget_frame.pack_forget()

    def toggle_popularity(self):
        """Show/hide popularity input fields"""
        if self.popularity_var.get():
            self.popularity_frame.pack(side=tk.LEFT, padx=10)
            self.popularity_from_label.pack(side=tk.LEFT, padx=5)
            self.popularity_min.pack(side=tk.LEFT, padx=5)
            self.popularity_to_label.pack(side=tk.LEFT, padx=5)
            self.popularity_max.pack(side=tk.LEFT, padx=5)
        else:
            self.popularity_frame.pack_forget()

    def toggle_date(self):
        """Show/hide date input fields"""
        if self.date_var.get():
            self.date_frame.pack(side=tk.LEFT, padx=10)
            self.date_from_label.pack(side=tk.LEFT, padx=5)
            self.date_min.pack(side=tk.LEFT, padx=5)
            self.date_to_label.pack(side=tk.LEFT, padx=5)
            self.date_max.pack(side=tk.LEFT, padx=5)
        else:
            self.date_frame.pack_forget()

    def toggle_genre(self):
        """Show/hide genre input fields"""
        if self.genre_var.get():
            self.genre_frame.pack(side=tk.LEFT, padx=10)
            self.genre_keywords_label.pack(side=tk.LEFT, padx=5)
            self.genre_keywords.pack(side=tk.LEFT, padx=5)
            self.neighbors_label.pack(side=tk.LEFT, padx=10)
            self.number_of_results_entry.pack(side=tk.LEFT, padx=5)
        else:
            self.genre_frame.pack_forget()

    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def search_movies(self):
        """Execute movie search in a separate thread"""
        self.search_btn.config(state='disabled', text="Searching...")
        self.update_status("Executing search...")

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

    def display_results(self, results):
        """Display search results in the table"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not results:
            self.update_status("No results found")
            messagebox.showinfo("No Results", "No movies found matching your criteria.")
        else:
            for row in results:
                # Extract data based on indices
                # Index 1: title, Index 3: origin_country, Index 8: budget
                # Index 11: popularity, Index 5: release_date, Index 6: genre_names, Index 9: revenue

                title = str(row[1]) if len(row) > 1 else "N/A"
                origin = str(row[3]).replace("['", "").replace("']", "") if len(row) > 3 else "N/A"
                budget = f"${float(row[8]):,.0f}" if len(row) > 8 else "$0"
                popularity = f"{float(row[11]):.2f}" if len(row) > 11 else "N/A"
                date = str(row[5]) if len(row) > 5 else "N/A"
                genre = str(row[6]) if len(row) > 6 else "N/A"
                revenue = f"${float(row[9]):,.0f}" if len(row) > 9 else "$0"

                # Insert into tree
                self.tree.insert('', tk.END, text=title,
                               values=(origin, budget, popularity, date, genre, revenue))

            self.update_status(f"Found {len(results)} movies")

        # Re-enable button
        self.search_btn.config(state='normal', text="Search")

    def show_error(self, error_msg):
        """Show error message"""
        self.update_status("Error during search")
        self.search_btn.config(state='normal', text="Search")
        messagebox.showerror("Error", f"An error occurred:\n{error_msg}")

    def sort_treeview_column(self, col, reverse):
        """Sort treeview content when a column header is clicked."""

        # Λίστα με όλα τα στοιχεία (tuples): (τιμή, k_id)
        l = []
        for k in self.tree.get_children(''):
            if col == '#0':
                # Η στήλη #0 (Title) παίρνεται με το .item(..., 'text')
                val = self.tree.item(k, 'text')
            else:
                # Οι υπόλοιπες στήλες παίρνονται με το .set(..., col)
                val = self.tree.set(k, col)
            l.append((val, k))

        # Συνάρτηση μετατροπής για σωστή σύγκριση (Handling numbers vs strings)
        def convert_value(value):
            # Καθαρισμός από $, κόμματα και κενά
            clean_val = str(value).replace('$', '').replace(',', '').strip()

            # Προσπάθεια μετατροπής σε αριθμό (float)
            try:
                return float(clean_val)
            except ValueError:
                # Αν δεν είναι αριθμός, επιστροφή ως κείμενο (πεζά για σωστή αλφαβητική σειρά)
                return clean_val.lower()

        # Ταξινόμηση της λίστας
        l.sort(key=lambda t: convert_value(t[0]), reverse=reverse)

        # Αναδιάταξη των items στο Treeview
        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)

        # Ενημέρωση του heading ώστε το επόμενο κλικ να κάνει την αντίστροφη ταξινόμηση
        self.tree.heading(col, command=lambda: self.sort_treeview_column(col, not reverse))


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieSearchGUI(root)
    root.mainloop()