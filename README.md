# City Transportation Connect

Determine which two cities are best to connect given how important you want the number of people connected or the distance between them

> GitHub repo: https://github.com/Nicholas-Polimeni/ml-hsr <br/>
> Hackathon project: https://devfolio.co/projects/city-transportation-connect-b0fc <br/>
> Hackathon submission date: Feb 11, 2024 <br/>
> Contributors: Faris Durrani, Nicholas Polimeni, Haruto Tanaka, Matthias Druhl

## The problem City Transportation Connect solves
> From the Devfolio project page

We want to connect two cities through some public transportation system. The initial idea is to find which city pair is best suited for a high-speed rail (HSR) connection but it expanded into encompassing all transportation systems for maximum flexibility.

Given the factors of the total population connected and the distance between the two cities, the algorithm will calculate the viability score, i.e., how socially cost-effective it is to connect the two cities given the population of the two metro areas and the distance between them.

We allow the user to adjust the importance of those two factors, call these coefficients A and B, through sliders from a selected source city as well as see the overall global picture of which two cities are most viable to be connected given those importance factors.

## How to run

1. Install the required packages (running Python 3.9)

```bash
pip install -r requirements.txt
```

2. Run the main file

```bash
streamlit run streamlit.py
```

3. Go to the link provided in the terminal

# License

This project is MIT licensed, as found in the [LICENSE](./LICENSE) file.

This project's documentation is Creative Commons licensed, as found in the [LICENSE-docs](./.github/LICENSE-docs) file.
