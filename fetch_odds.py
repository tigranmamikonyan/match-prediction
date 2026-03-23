import requests
import pandas as pd
from sqlalchemy import create_engine
import datetime

# --- CONFIGURATION ---
API_KEY = 'a24545cddc5f8d1babbdb2c40cfc3aa9'
REGIONS = 'eu' # 'eu' gives standard decimal odds from European bookies. 'us' or 'uk' work too.
MARKETS = 'totals' # 'totals' is the Over/Under market
ODDS_FORMAT = 'decimal'
DATE_FORMAT = 'iso'

# To protect your API quota, we loop through specific leagues. 
# You can add more from their documentation later!
LEAGUES = [
    "soccer_argentina_primera_division",
    "soccer_australia_aleague",
    "soccer_austria_bundesliga",
    "soccer_brazil_campeonato",
    "soccer_china_superleague",
    "soccer_conmebol_copa_libertadores",
    "soccer_conmebol_copa_sudamericana",
    "soccer_denmark_superliga",
    "soccer_efl_champ",
    "soccer_england_league1",
    "soccer_england_league2",
    "soccer_epl",
    "soccer_fa_cup",
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_qualifiers_europe",
    "soccer_fifa_world_cup_winner",
    "soccer_france_coupe_de_france",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_germany_bundesliga_women",
    "soccer_germany_dfb_pokal",
    "soccer_germany_liga3",
    "soccer_italy_serie_a",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_mexico_ligamx",
    "soccer_netherlands_eredivisie",
    "soccer_norway_eliteserien",
    "soccer_portugal_primeira_liga",
    "soccer_spain_copa_del_rey",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_spl",
    "soccer_sweden_allsvenskan",
    "soccer_switzerland_superleague",
    "soccer_uefa_champs_league",
    "soccer_uefa_champs_league_women",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_nations_league",
    "soccer_usa_mls"
]

# Database Connection
db_string = "postgresql://postgres:tiko400090@127.0.0.1:65432/football"
engine = create_engine(db_string)

all_odds_data = []

print("🌐 Fetching live odds from The Odds API...")

for league in LEAGUES:
    print(f"👉 Pulling {league}...")
    url = f'https://api.the-odds-api.com/v4/sports/{league}/odds'

    params = {
        'api_key': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': DATE_FORMAT,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"❌ Failed to get odds for {league}: {response.text}")
        continue

    odds_json = response.json()

    # --- PARSE THE JSON ---
    for match in odds_json:
        home_team = match['home_team']
        away_team = match['away_team']
        match_date = match['commence_time']

        # We need to dig into the bookmakers to find the Over 2.5 line
        over_25_price = None
        under_25_price = None

        # We will just grab the first available bookmaker's odds for simplicity (usually Pinnacle or Bet365)
        if len(match['bookmakers']) > 0:
            for market in match['bookmakers'][0]['markets']:
                if market['key'] == 'totals':
                    for outcome in market['outcomes']:
                        # We specifically want the 2.5 goal line
                        if outcome['name'] == 'Over' and outcome['point'] == 2.5:
                            over_25_price = outcome['price']
                        elif outcome['name'] == 'Under' and outcome['point'] == 2.5:
                            under_25_price = outcome['price']

        # Only save it if we successfully found a 2.5 line
        if over_25_price is not None:
            all_odds_data.append({
                'OddsApi_HomeTeam': home_team,
                'OddsApi_AwayTeam': away_team,
                'MatchDate': pd.to_datetime(match_date),
                'Over25_Odds': over_25_price,
                'Under25_Odds': under_25_price,
                'Bookmaker': match['bookmakers'][0]['title'],
                'League': league,
                'FetchedAt': pd.Timestamp.utcnow()
            })

# --- SAVE TO DATABASE ---
if len(all_odds_data) > 0:
    df_odds = pd.DataFrame(all_odds_data)

    print(f"\n✅ Successfully parsed {len(df_odds)} matches with Over/Under 2.5 odds!")

    try:
        # Save to a brand new table in Postgres
        df_odds.to_sql('Live_Odds', con=engine, if_exists='replace', index=False)
        print("💾 Odds successfully saved to the 'Live_Odds' table in PostgreSQL!")
    except Exception as e:
        print(f"❌ Database error: {e}")
else:
    print("📉 No Over 2.5 odds found right now.")