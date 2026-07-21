class Height:
    def __init__(self, min_height=1.50, max_height=2.20):
        self.min_height = min_height
        self.max_height = max_height

    def get_player_height(self):
        """Prompts user for input and ensures the height is within range."""
        while True:
            try: 
                height = float(input(f"Enter height for The player : "))
                if self.min_height <= height <= self.max_height:
                    return height
                else:
                    print(f"Invalid height! Please enter a value between {self.min_height}m and {self.max_height}m.")
            except ValueError:
                print("Invalid input! Please enter a numeric value.")
