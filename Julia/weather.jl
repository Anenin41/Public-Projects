# A simple program that fetches and displays weather information #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com
# Created: Tue 04 Mar 2025 @ 23:01:01 +0100
# Modified: Thu 06 Mar 2025 @ 13:57:42 +0100

# Packages
using HTTP, JSON3, DotEnv, ArgParse

# Load environmental variables
DotEnv.load!()

# Fetch API key
api_key = get(ENV, "OPENWEATHER_API_KEY", "")

# Check if the key loaded correctly
if isempty(api_key)
    error("API Key not found! Load the environmental variable.")
end

println("API Key Loaded: ", api_key)

# Function to parse command-line arguments for different cities
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--city", "-c"
        help = "City name for weather lookup"
        required = false
    end
    return parse_args(s)
end

# Get command line arguments
args = parse_commandline()

# Check Data Type and return relevant one depending on user prompt
if isnothing(args["city"])
    cities = ["Groningen", "Amsterdam", "Athens"]
else
    cities = args["city"]
end

# Function to export information from the API response
function fetch_info(city, weather_data)
    if haskey(weather_data, "main") && haskey(weather_data, "weather")
        temperature = weather_data["main"]["temp"]
        condition = weather_data["weather"][1]["description"]

        println("\nWeather in $city:")
        println("Temperature: $temperature °C")
        println("Condition: $condition")
    else
        println("Error fetching weather data:", weather_data)
    end
end

function main()
    if cities isa Vector
        for city in cities
            # Construct API request URL
            url = "http://api.openweathermap.org/data/2.5/weather?q=$city&appid=$api_key&units=metric"
            response = HTTP.get(url)
            weather_data = JSON3.read(response.body)
            fetch_info(city, weather_data)
        end
    else
        city = args["city"]
        # Construct API request URL
        url = "http://api.openweathermap.org/data/2.5/weather?q=$city&appid=$api_key&units=metric"
        response = HTTP.get(url)
        weather_data = JSON3.read(response.body)
        fetch_info(city, weather_data)
    end
end

# Fetch and print the Weather
main()
