# Publish this project on GitHub

You can delete this file after the project is on GitHub.

## 1. Personalise two files
- **README.md** – replace `YOUR-USERNAME` with your GitHub username (it appears in the badge and the clone command).
- **LICENSE** – replace `YOUR NAME` with your name.

## 2. Create an empty repository on GitHub
1. Sign in at https://github.com and select **New repository**.
2. Name it `house-price-prediction`, choose **Public**.
3. Leave **Add a README**, **.gitignore** and **license** unticked (the project already has them).
4. Select **Create repository**.

## 3. Upload the code

### Option A – from VS Code (no commands)
1. Open the project folder in VS Code.
2. Open the **Source Control** panel (Ctrl+Shift+G).
3. Select **Publish to GitHub**, sign in, choose *public repository*, keep all files ticked.

### Option B – from the terminal
```bash
git init
git add .
git commit -m "House price prediction: ML model, Flask app, tests"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/house-price-prediction.git
git push -u origin main
```
The first time, Git asks you to sign in to GitHub in the browser.

## 4. Check it worked
- Open the **Actions** tab on GitHub. The **CI** workflow trains the model and runs the tests; after
  2–3 minutes it should show a green tick, and the badge in the README turns green.
- On the repository page select the ⚙ next to **About** and add a description and topics such as
  `machine-learning`, `regression`, `scikit-learn`, `flask`, `python`, `house-price-prediction`.

## 5. Updating later
```bash
git add .
git commit -m "Describe what you changed"
git push
```
