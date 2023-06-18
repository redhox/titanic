import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
edition des variable
data = tableau deja remplit pour entrainé le model 
--------------------------------------------------
     PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0              1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1              2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2              3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S

test = tableau avec des information parcielle a traiter
-------------------------------------------------------
     PassengerId  Pclass                                          Name     Sex   Age  SibSp  Parch              Ticket      Fare Cabin Embarked
0            892       3                              Kelly, Mr. James    male  34.5      0      0              330911    7.8292   NaN        Q
1            893       3              Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0              363272    7.0000   NaN        S
2            894       2                     Myles, Mr. Thomas Francis    male  62.0      0      0              240276    9.6875   NaN        Q


"""
data = pd.read_csv("./assets/train.csv")
test = pd.read_csv("./assets/test.csv")
test_ids = test["PassengerId"]

"""
clean = 
    supression des colones Ticket PassengerId Name Cabin des tableau (axis=1 et la pour represanté les colones)
    prise en compte des collones que l'on garde et qui on des valeurs numerique 
    les colones que l'on garde on des element vide que l'on remplace pas la mediane des autre element de cette meme colone
    remplasement des element vide de la colone Embarked par une nouvelle valeurs "U" (valeurs non existante auparavant)
"""
def clean(data):
    data = data.drop(["Ticket", "PassengerId", "Name", "Cabin"], axis=1) 
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols:
        data[col].fillna(data[col].median(), inplace=True)
    data.Embarked.fillna("U", inplace=True)
    return data
data = clean(data)
test = clean(test)

"""
changement de valeur unique repeté dans les colones Sex et Embarked pour les changer en donner numerique
fit_transform() combine fit() et transform() en une seul comande,fit() est utilisée pour calculer et enregistrer tous les paramètres nécessaires à la transformation et transform() applique les paramètres pour transformer les donnees.
on dois utilisé fit() ou fit_transform() que sur les données d'entrainement. et uniquement transform() sur les donnees a transformer, comme ça la même transformation va etre apliquer sur les donner d'entrainement et de test.
"""
le = preprocessing.LabelEncoder()
columns = ["Sex", "Embarked"]
for col in columns:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    print(le.classes_)#stocke les classes trouvées par la méthode fit()
y = data["Survived"]
X = data.drop("Survived", axis=1)

"""
IA stuf
train_test_split = 
    les données X et les étiquettes y en ensembles d'apprentissage et de validation
    test_size:      taille du jeu de validation 0 = 1% , 1=100% des donner d'origine
    random_state:   caractère aléatoire de certaines fonctions ,peut être défini sur un entier, une instance RandomState ou None,est utilisé comme graine pour le générateur de nombres aléatoires si il est defini sur rien ="np.random 1" 
LogisticRegression = 
    régression logistique modèle linéaire de classification qui estime la probabilité qu'une instance appartienne à une classe particulière. Les probabilités sont ensuite utilisées pour faire des prédictions en choisissant la classe avec la probabilité la plus élevée.
    random_state:   etat aleatoire pour la reproductibiliter
    max_iter:       nombre maximum d'itérations
"""
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

"""
afichage des prediciton dans le terminal
"""
predictions = clf.predict(X_val)
accuracy_score(y_val, predictions)
print(accuracy_score(y_val, predictions))
"""
creation d'un dataframe avec comme premiere colone les id des passager sauvgardé avant le "clean" et la prediction fait par l'ia dans la deuxieme colones
puis finalement enregistre le dataframe dans un fichier csv
"""
submission_preds = clf.predict(test)
df = pd.DataFrame({"PassengerId": test_ids.values,
                   "Survived": submission_preds,
                  })
df.to_csv("./assets/submission.csv", index=False)
