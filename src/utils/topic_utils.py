from src.utils.core_utils import * 
import re
np.random.seed(42)

TOPICS_DIRECTORY = "/u/jai/efficiency/data/subreddit_topics/"
filename_to_topic_label = {
    "advice": "discussion_advice",
    "art": "hobbies/occupations_arts/writing",
    "cars": "hobbies/occupations_automotive",
    "discussion": "discussion_general",
    "diy": "hobbies/occupations_general",
    "education": "educational_general",
    "food": "lifestyle_food",
    "general_sports": "entertainment_sports",
    "location_reddits": "other_geography",
    "memes": "humor_memes/rage comics",
    "music": "entertainment_music",
    "nsfw": "nsfw",
    "politics": "other_news/politics",
    "scifi": "entertainment_genres",
    "tech": "technology_general",
    "universities": "other_universities",
    "video_games": "entertainment_video_games"
}



def generate_main_topic_mappings():
    """
    Link each subreddit to its main topic and subtopic, and each topic to all subtopics within it.
    Data is gathered from https://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits/ 
    updated as of April 2023.
    """
    with open(TOPICS_DIRECTORY + "main_subreddit_topics.txt", "r") as file:
        subreddit_topics = file.readlines()

    subreddit_to_subtopic = defaultdict(list)
    topic_to_subtopic = defaultdict(list)

    curr_topic = None
    curr_subtopic = None

    for topic in tqdm(subreddit_topics):
        topic = topic.strip().lower()
        if len(topic) < 3:
            continue

        # If the topic is a first-level heading denoted by a single #
        if re.match(r"^#\s?\*", topic):
            topic = re.findall(r"[^\*#]{2,}", topic)[0]
            curr_topic = topic
        
        # If the topic is a second-level heading denoted by a double ##
        elif re.match(r"^##\s?\*", topic):
            topic = re.findall(r"[^\*#]{2,}", topic)[0]
            curr_subtopic = topic
            topic_to_subtopic[curr_topic].append(curr_subtopic)
        
        # If the topic is a any lower, we group it into the higher-level topics
        elif re.match(r"^#{3,}\s?\*", topic):
            continue
        elif topic.startswith("**/r"):
            topic = topic.split(" ")[0]
            topic = topic[5:-2]
            subreddit_to_subtopic[topic].append(curr_topic + "_" + curr_subtopic)
        elif topic.startswith("/r"):
            topic = topic.split(" ")[0]
            topic = topic[3:]
            subreddit_to_subtopic[topic].append(curr_topic + "_" + curr_subtopic)
        else:
            print(topic)
    
    return subreddit_to_subtopic, topic_to_subtopic



def augment_subtopic_mappings(subreddit_to_subtopic, subreddits_of_interest):
    """
    The main subreddit list had links to additional Reddit pages with larger lists of 
    subreddits within a topic. We augment the main dictionary with this extra information.

    In addition, for all remaining subreddits that have not yet been accounted for,
    we check to see if they can be added to the discussion subtopic based on the
    subreddit name.
    """
    new_subreddits = 0
    
    # Add in all new subreddits scraped from additional links
    for file_name in os.listdir(TOPICS_DIRECTORY):
        with open(TOPICS_DIRECTORY + file_name, "r") as file:
            data = file.readlines()
        for line in data:
            line = line.strip().lower()
            if not line.startswith("http"):
                continue
            try:
                r_idx = line.index("/r/")
            except:
                continue
            sub = line[r_idx + 3:]
            subreddit_to_subtopic[sub].append(filename_to_topic_label[file_name[:-4]])
            new_subreddits += 1
            

    # For all existing subreddits, add an additional topic for discussion
    for sub in subreddit_to_subtopic:
        if sub.startswith("ask"):
            subreddit_to_subtopic[sub] = ["discussion_question/answer"] + subreddit_to_subtopic[sub]


    # For all subreddits that have not been represented, add an additional topic for discussion
    unrepresented_subreddits = list(set(subreddits_of_interest).difference(set(subreddit_to_subtopic.keys())))
    for subreddit in unrepresented_subreddits:
        if subreddit.startswith("ask"):
            subreddit_to_subtopic[subreddit] = ["discussion_question/answer"] + subreddit_to_subtopic[subreddit]
        elif "advice" in subreddit:
            subreddit_to_subtopic[subreddit] = ["discussion_advice"] + subreddit_to_subtopic[subreddit]
        elif "questions" in subreddit:
            subreddit_to_subtopic[subreddit] = ["discussion_question/answer"] + subreddit_to_subtopic[subreddit]

    return subreddit_to_subtopic



def select_subtopic_per_subreddit(subreddit_to_subtopic, subreddits_of_interest):
    """
    Map each subreddit to a unique subtopic for future analyses.
    """
    subreddit_to_single_subtopic = {}
    for subreddit in subreddits_of_interest:
        if len(subreddit_to_subtopic[subreddit]) > 0:
            subreddit_to_single_subtopic[subreddit] = np.random.choice(subreddit_to_subtopic[subreddit])
        else:
            subreddit_to_single_subtopic[subreddit] = 'no_topic_assigned'
    return subreddit_to_single_subtopic




def map_subtopic_to_topic(subreddit_to_subtopic, topic_to_subtopic):
    """
    Match each subtopic to the topic that it falls under. Many subtopics 
    """
    
    df =pd.DataFrame.from_dict(subreddit_to_subtopic, orient="index").reset_index()

    # We move "other" subtopics to the available larger topics
    # We also correct labels for subtopics not included in the topic_to_subtopic dictionary
    df =  df.replace({"other_geography": "locations", "other_news/politics": "politics", "entertainment_sports": "sports", 
                        "entertainment_video_games": "video_games", "educational_general": "education",  "educational_support": "education", 
                        "technology_self-improvement": "technology",  "animals_memes/rage comics": "animals", "animals": "animals", 
                        "general content_images": "general", "general content_gifs": "general",  "general content_videos": "general", 
                        "other_universities": "other"
                                           })

    education_subtopics = list(map(lambda x: "_".join(x), product(["educational"], topic_to_subtopic['educational'])))
    discussion_subtopics = list(map(lambda x: "_".join(x), product(["discussion"], topic_to_subtopic['discussion'])))
    entertainment_subtopics = list(map(lambda x: "_".join(x), product(["entertainment"], topic_to_subtopic['entertainment'])))
    lifestyle_subtopics = list(map(lambda x: "_".join(x), product(["lifestyle"], topic_to_subtopic['lifestyle'])))
    technology_subtopics = list(map(lambda x: "_".join(x), product(["technology"], topic_to_subtopic['technology'])))
    hobbies_subtopics = list(map(lambda x: "_".join(x), product(["hobbies/occupations"], topic_to_subtopic['hobbies/occupations'])))
    humor_subtopics = list(map(lambda x: "_".join(x), product(["humor"], topic_to_subtopic['humor'])))
    animals_subtopics = list(map(lambda x: "_".join(x), product(["animals"], topic_to_subtopic['animals'])))
    other_subtopics = list(map(lambda x: "_".join(x), product(["other"], topic_to_subtopic['other'])))


    df =  (pd.DataFrame(df).replace({sub: "education" for sub in education_subtopics})
            .replace({sub: "discussion" for sub in discussion_subtopics})
            .replace({sub: "entertainment" for sub in entertainment_subtopics})
            .replace({sub: "lifestyle" for sub in lifestyle_subtopics})
            .replace({sub: "technology" for sub in technology_subtopics})
            .replace({sub: "hobbies" for sub in hobbies_subtopics})
            .replace({sub: "humor" for sub in humor_subtopics})
            .replace({sub: "animals" for sub in animals_subtopics})
            .replace({sub: "other" for sub in other_subtopics})
        )



    return df.set_index("index")[0].to_dict()


if __name__ == "__main__":
    subreddits_of_interest = pd.read_csv() # Add in the name of the file that contains all subreddits you are examining
    subreddit_to_subtopic, topic_to_subtopic = generate_main_topic_mappings()
    subreddit_to_subtopic = augment_subtopic_mappings(subreddit_to_subtopic, subreddits_of_interest)
    subreddit_to_single_subtopic = select_subtopic_per_subreddit(subreddit_to_subtopic, subreddits_of_interest)
    subreddit_to_topic = map_subtopic_to_topic(subreddit_to_single_subtopic, topic_to_subtopic)
    
    ## Save the subreddit_to_topic dictionary
    
