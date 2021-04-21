from app import app
import cryptocompare
import subprocess
import tweepy
from urllib.request import urlopen
import json
from app import config



cryptocompare.cryptocompare._set_api_key_parameter(config.cryptoAPI)


url = config.steamurl
data = urlopen(url).read()


auth = tweepy.OAuthHandler(config.tweeterAPI_1, config.tweeterAPI_2)
auth.set_access_token(config.tweeterAPI_3, config.tweeterAPI_4)
api = tweepy.API(auth)
user = api.get_user('twitter')

command = '/home/delta/anaconda3/bin/gpustat --show-power'
def getgpupid():
    val = subprocess.run(command.split(' '), stdout=subprocess.PIPE).stdout.decode('utf-8')
    val = val.split('\n')
    val = val[1:]
    gpu0_val = val[0]
    gpu1_val = val[1]
    gpu0_val_ =gpu0_val.split('|')
    gpu1_val_ =gpu1_val.split('|')
    power0 = gpu0_val_[1]
    power0 = power0.split(',')[-1]
    power1 = gpu1_val_[1]
    power1 = power1.split(',')[-1]
    return power0,power1


@app.route('/')
def a():
	#a = cryptocompare.get_price('BTC', currency='USD', full=True)
	#print(a)
	return "hi"

@app.route('/btc')
def b():
	btc = cryptocompare.get_price('BTC', currency='AUD', full=True)
	price = btc['RAW']['BTC']['AUD']['PRICE']
	price = str(price)
	price = price + " AUD"
	if len(price) > 50:
		price = 'Error'
		return price
	return price

@app.route('/eth')
def c():
	btc = cryptocompare.get_price('ETH', currency='AUD', full=True)
	price = btc['RAW']['ETH']['AUD']['PRICE']
	price = str(price)
	price = price + " AUD"
	if len(price) > 50:
		price = 'Error'
		return price
	return price

@app.route('/power0')
def d():
	power0,_ = getgpupid()
	power0 = power0.lstrip(' ')
	if len(power0) > 50:
		return "Error"
	return power0

@app.route('/power1')
def e():
	_,power1= getgpupid()
	power1 = power1.lstrip(' ')
	if len(power1) > 50:
		return "Error"
	return power1

@app.route('/steam')
def f():
	data = urlopen(url).read()
	jsonData = json.loads(data)
	games = jsonData['response']['games']
	playtime = []
	for g in games:
		p = g['playtime_forever']
		if not isinstance(p,int):
			p = []
		playtime.append(p)

	total_playtime = sum(playtime)/60
	total_playtime = round(total_playtime,2)
	total_playtime = f'{total_playtime} hrs'
	if len(total_playtime) > 50:
		return "Error"
	return total_playtime

@app.route('/weather')
def weather(location='XXXXX'):
	response = urlopen(f'https://api.openweathermap.org/data/2.5/weather?q={location}&appid={config.weatherAPI}')
	response = response.read()
	response = json.loads(response)
	description = response['weather'][0]['description']
	temp  = response['main']['temp'] - 273.15
	temp = round(temp,2)

	return f'{description}, {temp} C'
