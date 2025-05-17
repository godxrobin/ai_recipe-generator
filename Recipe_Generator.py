import ast
import pandas as pd
import spacy
from fuzzywuzzy import fuzz

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class RecipeGenerator:
    def __init__(self, data_path='mini_recipes.csv'):
        self.df = pd.read_csv(data_path)
        # Convert string representations of lists to actual lists
        self.df['ingredients'] = self.df['ingredients'].apply(ast.literal_eval)
        self.df['steps'] = self.df['steps'].apply(ast.literal_eval)
    
    def extract_ingredients(self, user_input):
        """Extract ingredients from user input using spaCy"""
        doc = nlp(user_input.lower())
        ingredients = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                # Basic lemmatization
                ingredients.append(token.lemma_)
        
        return list(set(ingredients))  # Remove duplicates
    
    def find_recipes(self, user_ingredients, max_recipes=5):
        """Find recipes matching user ingredients"""
        matches = []
        
        for _, row in self.df.iterrows():
            match_score = self._calculate_match_score(user_ingredients, row['ingredients'])
            if match_score > 0:
                matches.append({
                    'id': row['id'],
                    'name': row['name'],
                    'time': row['minutes'],
                    'score': match_score,
                    'ingredients': row['ingredients'],
                    'steps': row['steps'],
                    'n_ingredients': row['n_ingredients']
                })
        
        # Sort by best match and return top results
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:max_recipes]
    
    def _calculate_match_score(self, user_ingredients, recipe_ingredients):
        """Calculate how well user ingredients match recipe ingredients"""
        score = 0
        recipe_ingredients_lower = [ing.lower() for ing in recipe_ingredients]
        
        for user_ing in user_ingredients:
            for recipe_ing in recipe_ingredients_lower:
                # Use fuzzy matching to account for variations
                if fuzz.partial_ratio(user_ing, recipe_ing) > 75:
                    score += 1
                    break
        
        return score

    def format_recipe(self, recipe):
        """Format a recipe for nice display"""
        output = f"\nRecipe: {recipe['name']} (ID: {recipe['id']})"
        output += f"\nTime: {recipe['time']} minutes"
        output += f"\n\nIngredients ({recipe['n_ingredients']}):"
        output += "\n" + "\n".join(f"- {ing}" for ing in recipe['ingredients'])
        output += "\n\nSteps:"
        output += "\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(recipe['steps']))
        return output

def main():
    generator = RecipeGenerator()
    
    print("=== Simple AI Recipe Generator ===")
    print("Example inputs: 'I have eggs and butter', 'chicken recipes', 'quick pasta'")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("\nWhat ingredients do you have? ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        ingredients = generator.extract_ingredients(user_input)
        print(f"\nLooking for recipes with: {', '.join(ingredients)}")
        
        recipes = generator.find_recipes(ingredients)
        
        if recipes:
            print(f"\nFound {len(recipes)} matching recipes:")
            for i, recipe in enumerate(recipes, 1):
                print(f"{i}. {recipe['name']} ({recipe['time']} min) - Score: {recipe['score']}/{len(ingredients)}")
            
            choice = input("\nEnter recipe number to view details, or 'back' to search again: ")
            if choice.isdigit() and 1 <= int(choice) <= len(recipes):
                print(generator.format_recipe(recipes[int(choice)-1]))
        else:
            print("\nNo recipes found with those ingredients. Try different ones.")

if __name__ == "__main__":
    main()