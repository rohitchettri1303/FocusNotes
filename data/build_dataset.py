import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Dataset configuration
# ============================================================

DATA_PER_CATEGORY = 8

AGENDAS = {
    "Website Development": {
        "PROBLEM": [
            "The website is loading very slowly.",
            "The homepage has performance issues.",
            "Users are experiencing slow page loading.",
            "The website has too many large images.",
            "The current page takes too long to load.",
            "The server response time is too high.",
            "The mobile version is performing poorly.",
            "The website has several broken links.",
        ],
        "IDEA": [
            "We could optimize the website images.",
            "Maybe we should use WebP images.",
            "We could introduce lazy loading.",
            "Perhaps a content delivery network would help.",
            "We could reduce the size of the JavaScript files.",
            "Maybe we should improve the caching strategy.",
            "We could optimize the mobile layout.",
            "Perhaps we should compress the static assets.",
        ],
        "ACTION": [
            "I'll investigate the image sizes.",
            "I'll check the server response time.",
            "I'll test WebP image support.",
            "I'll analyze the website performance.",
            "I'll check the broken links.",
            "I'll measure the mobile loading time.",
            "I'll test the caching configuration.",
            "I'll compare the page load times.",
        ],
        "DECISION": [
            "Let's use WebP for the website images.",
            "We decided to enable lazy loading.",
            "The team agreed to optimize the homepage.",
            "We've decided to improve the caching strategy.",
            "Let's move the static files to a CDN.",
            "We agreed to reduce the JavaScript bundle size.",
            "The team decided to fix the mobile layout.",
            "We've agreed to compress the website assets.",
        ],
        "INFORMATION": [
            "The homepage contains around fifty images.",
            "The average page load time is five seconds.",
            "The website currently uses PNG images.",
            "Most visitors access the site from mobile devices.",
            "The JavaScript bundle is approximately three megabytes.",
            "The server response time is currently two seconds.",
            "The website receives around ten thousand visitors daily.",
            "Several pages currently have broken links.",
        ],
        "IRRELEVANT": [
            "Did you watch the football match yesterday?",
            "What did you have for lunch?",
            "The weather is really nice today.",
            "Are you going to the movie tonight?",
            "I bought a new pair of shoes yesterday.",
            "What time are you leaving for home?",
            "Did you see the new restaurant nearby?",
            "I need to call my friend later.",
        ],
    },

    "College Project": {
        "PROBLEM": [
            "Our dataset contains too many missing values.",
            "The model is giving poor predictions.",
            "We don't have enough training data.",
            "The classes in our dataset are imbalanced.",
            "Our current accuracy is too low.",
            "The preprocessing step is producing errors.",
            "Several records contain incorrect values.",
            "The model is taking too long to train.",
        ],
        "IDEA": [
            "We could try KNN imputation.",
            "Maybe we should collect more data.",
            "We could compare several machine learning models.",
            "Perhaps we should use feature selection.",
            "We could try balancing the classes.",
            "Maybe a transformer model would improve the results.",
            "We could experiment with different preprocessing methods.",
            "Perhaps we should engineer additional features.",
        ],
        "ACTION": [
            "I'll clean the missing values.",
            "I'll compare the classification models.",
            "I'll collect additional training samples.",
            "I'll evaluate the class distribution.",
            "I'll calculate the F1 score for each model.",
            "I'll test different preprocessing techniques.",
            "I'll prepare the dataset for training.",
            "I'll analyze the model errors.",
        ],
        "DECISION": [
            "We decided to use F1 score as our main metric.",
            "Let's use Random Forest as the baseline model.",
            "The team agreed to collect more samples.",
            "We've decided to use cross validation.",
            "We agreed to remove the duplicate records.",
            "Let's use the cleaned dataset for training.",
            "The team decided to compare three models.",
            "We've agreed to document the preprocessing steps.",
        ],
        "INFORMATION": [
            "The dataset contains five thousand records.",
            "There are twelve features in the dataset.",
            "The target variable contains three classes.",
            "Around ten percent of the records have missing values.",
            "The current model accuracy is seventy percent.",
            "The project has three machine learning models.",
            "The training dataset contains four thousand samples.",
            "The test dataset contains one thousand samples.",
        ],
        "IRRELEVANT": [
            "What are you doing this weekend?",
            "Did you watch the match last night?",
            "I need to buy groceries after class.",
            "The cafeteria is crowded today.",
            "What time does the bus leave?",
            "I watched a great movie yesterday.",
            "Are you joining us for dinner?",
            "The weather has been strange today.",
        ],
    },

    "Financial Analysis": {
        "PROBLEM": [
            "Operating costs have increased significantly.",
            "Revenue has declined this quarter.",
            "The company is spending too much on marketing.",
            "Profit margins are lower than expected.",
            "Several expenses are above budget.",
            "Cash flow has become unstable.",
            "The sales numbers are below target.",
            "The cost of operations is increasing.",
        ],
        "IDEA": [
            "We could reduce unnecessary expenses.",
            "Maybe we should review the marketing budget.",
            "We could improve the pricing strategy.",
            "Perhaps we should focus on the most profitable products.",
            "We could reduce operational costs.",
            "Maybe we should analyze regional sales.",
            "We could improve our forecasting model.",
            "Perhaps we should review supplier costs.",
        ],
        "ACTION": [
            "I'll analyze the regional revenue.",
            "I'll review the marketing expenses.",
            "I'll calculate the profit margin.",
            "I'll compare this quarter with last quarter.",
            "I'll investigate the increase in operating costs.",
            "I'll prepare the monthly financial report.",
            "I'll check the sales performance by region.",
            "I'll analyze the cash flow statement.",
        ],
        "DECISION": [
            "We decided to reduce the marketing budget.",
            "Let's review the pricing strategy next month.",
            "The team agreed to reduce operational expenses.",
            "We've decided to focus on the profitable products.",
            "We agreed to revise the sales targets.",
            "Let's increase the forecasting frequency.",
            "The management decided to review supplier contracts.",
            "We've agreed to monitor cash flow weekly.",
        ],
        "INFORMATION": [
            "Revenue increased by twelve percent last year.",
            "The current operating margin is eighteen percent.",
            "Marketing expenses account for twenty percent of costs.",
            "The company generated ten million in revenue.",
            "Sales are highest in the southern region.",
            "Operating costs increased by eight percent.",
            "The company has three major revenue streams.",
            "The current budget covers the next twelve months.",
        ],
        "IRRELEVANT": [
            "Did you watch the football match?",
            "What should we have for lunch?",
            "I bought a new phone yesterday.",
            "Are you going to the party?",
            "The weather is very hot today.",
            "What time is the movie?",
            "My friend is visiting tomorrow.",
            "I need to get some groceries.",
        ],
    },

    "Marketing Campaign": {
        "PROBLEM": [
            "The campaign is generating very few leads.",
            "Our conversion rate is too low.",
            "The advertisement is not reaching enough people.",
            "Customer engagement has declined.",
            "The campaign is exceeding its budget.",
            "The email open rate is below target.",
            "Our social media engagement is falling.",
            "The current campaign is not generating enough sales.",
        ],
        "IDEA": [
            "We could test a different advertisement.",
            "Maybe we should target a younger audience.",
            "We could experiment with video content.",
            "Perhaps we should change the email subject lines.",
            "We could increase our social media presence.",
            "Maybe we should test a different landing page.",
            "We could introduce a referral campaign.",
            "Perhaps influencer marketing could help.",
        ],
        "ACTION": [
            "I'll analyze the campaign conversion rate.",
            "I'll test the new advertisement.",
            "I'll review the audience demographics.",
            "I'll compare the email open rates.",
            "I'll analyze the social media engagement.",
            "I'll prepare a new landing page.",
            "I'll check the campaign spending.",
            "I'll review the customer acquisition data.",
        ],
        "DECISION": [
            "We decided to target a younger audience.",
            "Let's launch the new advertisement.",
            "The team agreed to test video content.",
            "We've decided to change the email strategy.",
            "We agreed to increase social media activity.",
            "Let's introduce the referral campaign.",
            "The marketing team decided to test influencers.",
            "We've agreed to change the landing page.",
        ],
        "INFORMATION": [
            "The campaign reached fifty thousand users.",
            "The current conversion rate is three percent.",
            "Email open rates are currently twenty percent.",
            "Social media engagement increased last month.",
            "The campaign budget is one hundred thousand.",
            "Most customers are between twenty and thirty years old.",
            "The campaign generated five hundred leads.",
            "The advertisement has been running for two weeks.",
        ],
        "IRRELEVANT": [
            "Did you watch the match?",
            "What are you doing after work?",
            "I need to buy a new shirt.",
            "The restaurant nearby is really good.",
            "What time should we leave?",
            "The weather looks cloudy today.",
            "Did you call your parents?",
            "I'm planning a trip next month.",
        ],
    },

    "Software Development": {
        "PROBLEM": [
            "The application crashes during login.",
            "The API response time is too slow.",
            "The database queries are inefficient.",
            "Several bugs remain unresolved.",
            "The application consumes too much memory.",
            "The current authentication system is unreliable.",
            "The deployment process frequently fails.",
            "The codebase has many duplicate functions.",
        ],
        "IDEA": [
            "We could optimize the database queries.",
            "Maybe we should introduce caching.",
            "We could refactor the authentication module.",
            "Perhaps we should use automated testing.",
            "We could improve the deployment pipeline.",
            "Maybe we should introduce a logging system.",
            "We could split the application into smaller services.",
            "Perhaps we should improve the error handling.",
        ],
        "ACTION": [
            "I'll investigate the login crash.",
            "I'll profile the database queries.",
            "I'll fix the authentication issue.",
            "I'll add automated tests.",
            "I'll review the deployment pipeline.",
            "I'll analyze the memory usage.",
            "I'll refactor the duplicate functions.",
            "I'll investigate the API response time.",
        ],
        "DECISION": [
            "We decided to introduce caching.",
            "Let's refactor the authentication module.",
            "The team agreed to add automated testing.",
            "We've decided to improve error handling.",
            "We agreed to optimize the database queries.",
            "Let's introduce centralized logging.",
            "The team decided to split the application into services.",
            "We've agreed to improve the deployment pipeline.",
        ],
        "INFORMATION": [
            "The application currently has three services.",
            "The API receives around ten thousand requests daily.",
            "The database contains two million records.",
            "The current application uses Python.",
            "The deployment pipeline runs every evening.",
            "The application has around fifty thousand users.",
            "The backend uses a relational database.",
            "The project currently has twenty developers.",
        ],
        "IRRELEVANT": [
            "Did you watch the movie yesterday?",
            "What are you eating for lunch?",
            "I need to buy groceries.",
            "Are you going to the gym today?",
            "The weather is great today.",
            "What time is dinner?",
            "My friend is coming tomorrow.",
            "Did you see the new restaurant?",
        ],
    },

    "Product Development": {
        "PROBLEM": [
            "Customers are unhappy with the current product.",
            "The product has a high return rate.",
            "Users are struggling with the onboarding process.",
            "The current design is confusing.",
            "Customer retention has declined.",
            "The product is too expensive for some users.",
            "Several important features are missing.",
            "The latest release introduced usability problems.",
        ],
        "IDEA": [
            "We could simplify the onboarding process.",
            "Maybe we should introduce a free trial.",
            "We could redesign the main dashboard.",
            "Perhaps we should add more customization options.",
            "We could improve the search functionality.",
            "Maybe we should introduce personalized recommendations.",
            "We could add a customer feedback feature.",
            "Perhaps we should improve the mobile experience.",
        ],
        "ACTION": [
            "I'll analyze the customer feedback.",
            "I'll review the onboarding process.",
            "I'll investigate the return rate.",
            "I'll prepare a dashboard redesign.",
            "I'll analyze customer retention.",
            "I'll review the missing features.",
            "I'll test the new search functionality.",
            "I'll compare user behavior across devices.",
        ],
        "DECISION": [
            "We decided to simplify onboarding.",
            "Let's introduce a free trial.",
            "The team agreed to redesign the dashboard.",
            "We've decided to add personalization.",
            "We agreed to improve the mobile experience.",
            "Let's add the customer feedback feature.",
            "The product team decided to improve search.",
            "We've agreed to prioritize the missing features.",
        ],
        "INFORMATION": [
            "The product has fifty thousand active users.",
            "Customer retention is currently seventy percent.",
            "The current return rate is five percent.",
            "Most users access the product through mobile devices.",
            "The product has been available for three years.",
            "The latest release was launched last month.",
            "Customers have requested twenty new features.",
            "The product currently has four subscription plans.",
        ],
        "IRRELEVANT": [
            "Did you watch the football match?",
            "What did you eat for breakfast?",
            "I need to buy a new laptop.",
            "Are you going out tonight?",
            "The weather is beautiful today.",
            "What time is the meeting?",
            "I watched a movie yesterday.",
            "My friend is visiting tomorrow.",
        ],
    },

    "Event Planning": {
        "PROBLEM": [
            "The venue does not have enough seating.",
            "Several speakers have not confirmed attendance.",
            "The event budget is too high.",
            "We have not sold enough tickets.",
            "The registration process is confusing.",
            "The catering company has limited availability.",
            "The event schedule has several conflicts.",
            "The current promotion is not attracting enough people.",
        ],
        "IDEA": [
            "We could move the event to a larger venue.",
            "Maybe we should invite another speaker.",
            "We could introduce early bird tickets.",
            "Perhaps we should simplify registration.",
            "We could increase social media promotion.",
            "Maybe we should adjust the event schedule.",
            "We could offer group discounts.",
            "Perhaps we should change the catering package.",
        ],
        "ACTION": [
            "I'll contact the larger venue.",
            "I'll confirm the speaker availability.",
            "I'll review the event budget.",
            "I'll analyze ticket sales.",
            "I'll check the registration process.",
            "I'll contact the catering company.",
            "I'll update the event schedule.",
            "I'll prepare a new promotional plan.",
        ],
        "DECISION": [
            "We decided to move to a larger venue.",
            "Let's introduce early bird tickets.",
            "The team agreed to simplify registration.",
            "We've decided to increase promotion.",
            "We agreed to offer group discounts.",
            "Let's change the catering package.",
            "The organizers decided to adjust the schedule.",
            "We've agreed to invite another speaker.",
        ],
        "INFORMATION": [
            "The event currently has three hundred registrations.",
            "The venue can accommodate five hundred people.",
            "The event is scheduled for next month.",
            "Four speakers have confirmed attendance.",
            "The current event budget is two hundred thousand.",
            "Ticket sales increased last week.",
            "The event has six sessions.",
            "The registration deadline is next Friday.",
        ],
        "IRRELEVANT": [
            "Did you watch the match yesterday?",
            "What are you doing tonight?",
            "I need to buy some clothes.",
            "The restaurant was excellent.",
            "The weather is hot today.",
            "What time should we have lunch?",
            "Did you call your friend?",
            "I'm planning a holiday next month.",
        ],
    },

    "Team Management": {
        "PROBLEM": [
            "The team is missing several deadlines.",
            "Communication between departments is poor.",
            "Some tasks are not clearly assigned.",
            "The workload is unevenly distributed.",
            "Team meetings are becoming unproductive.",
            "Several employees are unclear about their responsibilities.",
            "Project updates are not being communicated properly.",
            "The team is struggling to prioritize tasks.",
        ],
        "IDEA": [
            "We could introduce weekly progress meetings.",
            "Maybe we should use a shared task board.",
            "We could redistribute the workload.",
            "Perhaps we should define responsibilities more clearly.",
            "We could introduce shorter meetings.",
            "Maybe we should create a communication channel.",
            "We could prioritize tasks using deadlines.",
            "Perhaps we should introduce regular feedback sessions.",
        ],
        "ACTION": [
            "I'll review the current task assignments.",
            "I'll create a shared task board.",
            "I'll analyze the workload distribution.",
            "I'll schedule a progress meeting.",
            "I'll document everyone's responsibilities.",
            "I'll review the project communication process.",
            "I'll prepare a prioritized task list.",
            "I'll collect feedback from the team.",
        ],
        "DECISION": [
            "We decided to use a shared task board.",
            "Let's introduce weekly progress meetings.",
            "The team agreed to redistribute the workload.",
            "We've decided to clarify responsibilities.",
            "We agreed to shorten the weekly meetings.",
            "Let's create a dedicated communication channel.",
            "The management team decided to prioritize urgent tasks.",
            "We've agreed to introduce monthly feedback sessions.",
        ],
        "INFORMATION": [
            "The team currently has fifteen members.",
            "There are four departments involved in the project.",
            "The project deadline is next month.",
            "The team currently has twenty open tasks.",
            "Weekly meetings are held every Monday.",
            "Three members are working on the backend.",
            "The project has five major milestones.",
            "Most tasks are tracked using a shared system.",
        ],
        "IRRELEVANT": [
            "Did you watch the football match?",
            "What did you have for lunch?",
            "I'm going to the cinema tonight.",
            "The weather is really good today.",
            "I bought a new pair of shoes.",
            "What time are you leaving?",
            "Did you see the new restaurant?",
            "I'm planning a trip next month.",
        ],
    },
}


# ============================================================
# Load the master dataset
# ============================================================

INPUT_FILE = "data/conversation_dataset_v2.csv"

df = pd.read_csv(INPUT_FILE)

# Basic validation
required_columns = ["agenda", "sentence", "relevance", "category"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Remove empty rows
df = df.dropna(subset=required_columns)

# Remove duplicate sentences within the same agenda
df = df.drop_duplicates(
    subset=["agenda", "sentence"]
).reset_index(drop=True)

# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# Train / Test split
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["category"]
)


# ============================================================
# Save datasets
# ============================================================

train_df.to_csv(
    "data/training_dataset.csv",
    index=False
)

test_df.to_csv(
    "data/test_dataset.csv",
    index=False
)


# ============================================================
# Dataset information
# ============================================================

print()
print("=" * 50)
print("FOCUSNOTES DATASET")
print("=" * 50)

print(f"Total examples: {len(df)}")
print(f"Training examples: {len(train_df)}")
print(f"Testing examples: {len(test_df)}")

print()
print("Category distribution:")
print(df["category"].value_counts())

print()
print("Agenda distribution:")
print(df["agenda"].value_counts())

print()
print("Files created:")
print("data/training_dataset.csv")
print("data/test_dataset.csv")