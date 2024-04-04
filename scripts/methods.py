"""This scrip contains functions to extract data points and create visualisation from survey data"""

from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from deep_translator import GoogleTranslator
from scipy.stats.mstats import winsorize


def plot_counts(df, column, title, custom_order=None, showticklabels=True, show_fig=True, additional_data_points=None, translate=True, multiple_choice=False, linkert=False):
    """Plots counts of options selected by respondents for the selected question

    Args:
        df (_type_): DataFrame containing survey results
        column (_type_): Question number for which results need to be plot
        title (_type_): Short description of the survey question
        custom_order (_type_, optional): Odered list of options. Defaults to None.
        showticklabels (bool, optional): Set false for switching off ticklabels on x-axis. Defaults to True.
        show_fig (bool, optional): Displays figure. Defaults to True.
        additional_data_points (bool, optional): Dictionary of additional points. Defaults to False.
        translate (bool, optional): Set False if options don't need to be translated. Defaults to True.
    """
    # Calculate counts of each response
    if multiple_choice:
        response_counts = df[column].str.split(
            ',').explode().value_counts().reset_index(drop=False)
    else:
        response_counts = df[column].value_counts().reset_index(drop=False)

    response_counts.columns = [column, 'Count']
    response_counts[column] = response_counts[column].str.rstrip()

    if additional_data_points:
        additional_data_points = pd.DataFrame(
            additional_data_points,  index=[0]).T.reset_index()
        additional_data_points.rename(
            columns={'index': column, 0: 'Count'}, inplace=True)
        # additional_data_points[field] = additional_data_points[column]
        response_counts = pd.concat(
            [response_counts, additional_data_points], ignore_index=True)

    if custom_order:
        response_counts[column] = pd.Categorical(
            response_counts[column], categories=custom_order, ordered=True)
        response_counts = response_counts.sort_values(by=column)

    if translate:
        response_counts[column] = response_counts[column].apply(
            translate_to_english)

    # Plot using Matplotlib
    plt.figure(figsize=(10, 6))
    num_categories = len(response_counts[column])
    if linkert:
        colors = plt.cm.RdYlGn(np.arange(num_categories))
    else:
        if num_categories <= 10:
            colors = plt.cm.tab10(np.arange(num_categories))  # Generate colors
        else:
            colors = plt.cm.tab20(np.arange(num_categories))
    bars = plt.bar(response_counts[column], response_counts['Count'],
                   color=colors, label=response_counts[column])  # Use colors for bars
    plt.title(title.upper())

    # Show the tick labels based on the showticklabels parameter
    if not showticklabels:
        plt.xticks([])
        plt.xlabel('')
        # Add color legends for each bar
        legend_labels = response_counts[column]
        for bar, label in zip(bars, legend_labels):
            bar.set_label(label)
        # Display legend
        plt.legend(bbox_to_anchor=(0.02, -0.02), loc='upper left',
                   borderaxespad=0.0, mode="expand")

    else:
        plt.xlabel(title)
        plt.xticks(rotation=45, ha='right')

    plt.ylabel('Count')
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(f"../figures/{column}_{title}.jpeg")

    if show_fig:
        # Show the plot
        plt.show()


def plot_ranks(df, options, title, label, show_fig=True,):
    """
    Plot stacked horizontal bar chart showing ranks of options.

    Parameters:
    - df (DataFrame): Input DataFrame containing ranks.
    - options (dict): Dictionary mapping option numbers to corresponding labels.
    - title (str): Title of the plot.
    - label (str): Label for the plot/question.
    - show_fig (bool, optional): Whether to display the plot (default is True).

    Returns:
    - None
    """
    # Rename columns of the DataFrame
    df_melted = df.rename(columns=options)

    # Melt the DataFrame to long format
    df_melted = df_melted[options.values()].melt(
        var_name='Option', value_name='Rank')

    # Group by Option and Rank, then count occurrences and reshape
    df_melted = df_melted.groupby(
        ['Option', 'Rank']).size().unstack().reset_index()

    # Rename columns for better readability
    df_melted = df_melted.rename(
        columns={'1': 'Rank 1', '2': 'Rank 2', '3': 'Rank 3'})

    # Calculate the sum of ranks and sort by the sum
    df_melted['sum'] = df_melted[['Rank 1', 'Rank 2', 'Rank 3']].sum(axis=1)
    df_melted = df_melted.sort_values(
        'sum', ascending=True).set_index('Option')

    # Select only the columns for Rank 1, Rank 2, and Rank 3
    df_melted = df_melted[['Rank 1', 'Rank 2', 'Rank 3']]

    # Plot the stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    df_melted.plot(kind='barh', stacked=True, ax=ax)

    # Set y-axis tick labels alignment
    ax.set_yticklabels(ax.get_yticklabels(), ha='right')

    # Turn off y-axis labels
    ax.set_ylabel('')
    ax.set_xlabel('Count')

    # Turn off top and right box borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move legend to the bottom right corner
    ax.legend(loc='lower right')

    # Set title and labels
    ax.set_title(title.upper())

    # Show the plot
    if show_fig:
        plt.show()

    fig.savefig(f'../figures/{label}_{title}.png', bbox_inches='tight')


def translate_to_english(text):
    """
    Translate text from Italian to English.

    Parameters:
    - text (str): Input text in Italian.

    Returns:
    - str: Translated text in English.
    """
    translation = GoogleTranslator(source='it', target='en').translate(text)
    return translation


def plot_policy_solutions(df, title, label):
    """
    Plot the policy solutions chosen by respondents.

    Parameters:
    - df (DataFrame): DataFrame containing survey responses.
    - title (str): Title for the plot.
    - label (str): Label for saving the plot.

    Returns:
    - None
    """
    columns = ['Q19_1', 'Q19_2', 'Q19_3', 'Q19_4', 'Q19_5', 'Q19_6']

    options = [
        'Migliorare le infrastrutture critiche (drenaggio potenziato, barriere contro le inondazioni, serbatoi)',
        'Uso sostenibile del territorio (evitare di costruire e coltivare in aree a rischio di inondazioni, riforestare)',
        'Sistemi di allarme rapido (allarmi tempestive e precisi)',
        'Coinvolgimento della comunità (coinvolgimento della popolazione locale nella preparazione alle alluvioni)',
        'Politiche governative (priorità alla mitigazione delle inondazioni, norme urbanistiche)',
        'Tassare le industrie inquinanti',
    ]

    options = [translate_to_english(x) for x in options]

    options = dict(zip(columns, options))

    df_melted = df.rename(columns=options)
    df_melted = df_melted[options.values()]
    # Melt the DataFrame to long format
    df_melted = df_melted[options.values()].melt(
        var_name='Option', value_name='Choice')
    # Group by Option and Rank, then count occurrences and reshape
    df_melted = df_melted.groupby(
        ['Option', 'Choice']).size().unstack().reset_index()
    df_melted.rename(columns={'Option': 'Policy',
                              'Sicuramente': 'Definitely',
                              'Forse': 'Maybe',
                              'Per niente': 'Not at all'}, inplace=True)
    df_melted = df_melted[['Policy', 'Definitely', 'Maybe', 'Not at all']]
    df_melted = df_melted.sort_values(['Definitely', 'Maybe'])

    # Set 'Policy' column as the index
    df_melted.set_index('Policy', inplace=True)

    # Plot the stacked horizontal bar chart with custom colors
    fig, ax = plt.subplots(figsize=(8, 6))
    df_melted.plot(kind='barh', stacked=True, ax=ax,
                   color=['green', 'orange', 'red'])

    # Set y-axis tick labels alignment
    ax.set_yticklabels(ax.get_yticklabels(), ha='right')

    # Turn off top and right box borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set title and labels
    ax.set_title("Policy Choices")
    ax.set_xlabel("Count")

    # Move legend to the bottom right corner
    ax.legend(loc='upper left', bbox_to_anchor=(-1, 0.1))

    # Show the plot
    plt.show()

    fig.savefig(f'../figures/{label}_{title}.png', bbox_inches='tight')


def plot_linkert_scale(df, title, label, columns, choices, show_fig=True, cmap='RdYlGn', sort=False):
    """
    Plot the Likert scale responses for each statement.

    Parameters:
    - df (DataFrame): DataFrame containing survey responses.
    - title (str): Title for the plot.
    - label (str): Label for saving the plot.
    - columns (dict): Dictionary mapping column names to statement descriptions.
    - choices (list): List of choices in the Likert scale.
    - show_fig (bool, optional): Whether to display the plot. Default is True.
    - cmap (str, optional): Name of the colormap to use for the plot. Default is 'RdYlGn'.
    - sort (bool, optional): Whether to sort the statements by the mean score. Default is False.

    Returns:
    - None
    """
    df_melted = df[columns.keys()].rename(columns=columns)
    df_melted = df_melted[columns.values()].melt(
        var_name='Satement', value_name='Choice')
    df_melted = df_melted.groupby(
        ['Satement', 'Choice']).size().unstack().reset_index()
    df_melted.set_index('Satement', inplace=True)
    df_melted = df_melted[choices]
    choices_dict = dict(
        zip(choices, [translate_to_english(x) for x in list(df_melted.columns)]))
    df_melted = df_melted.rename(columns=choices_dict)

    if sort:
        df_melted = df_melted.sort_values(
            list(choices_dict.values()), ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    df_melted.plot(ax=ax, kind='barh', stacked=True, cmap=cmap)

    # Turn off y-axis labels
    ax.set_ylabel('')
    ax.set_xlabel('Count')

    # Turn off top and right box borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set title and labels
    ax.set_title(title.upper())

    ax.legend(bbox_to_anchor=(-0.55, -0.15), loc='upper left',
              borderaxespad=0.0, ncol=len(choices))

    # Show the plot
    if show_fig:
        plt.show()

    fig.savefig(f'../figures/{label}_{title}.png', bbox_inches='tight')


def plot_response_sankey():
    # Define the source and target nodes for each flow
    source = [0, 1, 2, 3, 4, 1, 2, 3, 4]
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    value = [624, 618, 562, 505, 500, 6, 56, 57, 5]

    label = [
        "Started Survey (624)",  # Node 0
        "Resident of Emilia Romagna (618)",  # Node 1
        "1st Attention Test (562)",  # Node 2
        "2nd Attention Test (505)",  # Node 3
        "Finished Survey (500)",  # Node 4
        "",
        "Not a Resident (6)",  # Node 5
        "Failed Attention Test 1 (56)",  # Node 6
        "Failed Attention Test 2 (57)",  # Node 7
        "Dropped Out (5)",  # Node 8
    ]

    # Define the colors for each flow
    colors = [
        "rgba(31, 119, 180, 0.8)",
        "rgba(255, 127, 14, 0.8)",
        "rgba(44, 160, 44, 0.8)",
        "rgba(214, 39, 40, 0.8)",
        "rgba(148, 103, 189, 0.8)",
        "rgba(140, 86, 75, 0.8)",
        "rgba(227, 119, 194, 0.8)",
        "rgba(127, 127, 127, 0.8)",
        "rgba(188, 189, 34, 0.8)"
    ]

    # Create the Sankey diagram object
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=label,
            color=colors
        ),
        link=dict(
            # indices correspond to labels, eg A1, A2, A1, B1, ...
            source=source,
            target=target,
            value=value
        ))])

    # Set the title and font size for the diagram
    fig.update_layout(
        title_text="Sankey Diagram of Survey Responses", font_size=10)

    # Show the diagram
    fig.write_html("../figures/sanky.html")
    fig.show()


def plot_survey_duration(duration):
    # Create a box plot to visualize the survey duration
    plt.figure(figsize=(8, 6))
    plt.boxplot([duration], labels=['Duration'])
    plt.title('Box plot of time taken to complete the survey')
    plt.ylabel('Minutes')
    plt.xlabel('Time taken to complete the survey')
    plt.grid(True)
    plt.savefig('../figures/completion_time.png')
    plt.show()


def plot_pie_chart(regions):
    plt.figure(figsize=(6, 6), )
    plt.pie(regions['count'], labels=regions['province'],
            autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Survey responses received from different provinces in Emilia-Romagna')
    plt.tight_layout()
    plt.savefig("../figures/response_from_region.png")
    plt.show()
