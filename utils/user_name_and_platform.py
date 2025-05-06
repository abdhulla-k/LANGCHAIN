def ask_for_platform():
    social_media_platforms = [
        "Linkedin",
        "Facebook",
        "Instagram",
        "Twitter",
        "Snapchat",
    ]

    social_media: int = int(
        input(
            "\nChoose social media platform:  \nEnter 1 for Linkedin, 2 for Facebook, 3 for Instagram, 4 for Twitter, 5 for Snapchat: "
        )
    )
    if social_media <= 5:
        print("You have selected: ", social_media_platforms[social_media - 1])
        return social_media_platforms[social_media - 1]
    else:
        print("Invalid choice. Please select a number between 1 and 5. \n\n")
        return ask_for_platform()
