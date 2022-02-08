# Import pandas
import pandas as pd

# Read in dataset
transfusion = pd.read_csv('transfusion.data')

# Print out the first rows of our dataset
transfusion.head()

#transfusion.info()

# Rename target column as 'target' for brevity 
transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True
)

# Print out the first 2 rows
print(transfusion.head(2))
# Print target incidence proportions, rounding output to 3 decimal places
print(transfusion.target.value_counts(normalize=True).round(3))

# Import train_test_split method
from sklearn.model_selection import train_test_split

# Split transfusion DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion.target
)


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


def predictResult(d):
    ## 2                  5                   1250             16
    df = pd.DataFrame(data=d)
    y_pred=logreg.predict(df)
    print("prediction")
    print(y_pred)




from flask import Flask, request, render_template, Markup, send_from_directory, session, flash, Response
import socket
app = Flask(__name__)

app.config["DEBUG"] = True


app.route('/')
def homePage():
    
        return render_template("home.html")

@app.route('/postWeatherData', methods=['GET', 'POST'])
def postWeatherData():
    if request.method == 'GET':
        r = request.args.get('Recency')
        f = request.args.get('Frequency')
        m = request.args.get('Monetary')
        t = request.args.get('Time')
        d = {'Recency (months)': [r], 'Frequency (times)': [f], 'Monetary (c.c. blood)': [m], 'Time (months)': [t]}

    return predictResult(d)

if __name__ == '__main__':

    s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    s.connect(('8.8.8.8',80))
    localhost=s.getsockname()[0]
    app.run(host = localhost, port = 8000)
