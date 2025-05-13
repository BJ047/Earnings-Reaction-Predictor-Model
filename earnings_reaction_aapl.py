import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def get_aapl_features(start_date):
    ticker = "AAPL"
    t = yf.Ticker(ticker)
    er = t.get_earnings_dates(limit=100)
    if er is None or er.empty:
        raise ValueError("No earnings data returned for AAPL.")
    er.index = er.index.tz_localize(None)
    er = er[er.index >= start_date]
    if er.empty:
        raise ValueError("No earnings in the specified date range.")

    hist = t.history(
        start=start_date - dt.timedelta(days=30),
        end=dt.datetime.today() + dt.timedelta(days=1)
    )
    if hasattr(hist.index, 'tz') and hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    hist['ret'] = hist['Close'].pct_change()

    rows = []
    for date, row in er.iterrows():
        est = row['EPS Estimate']
        act = row.get('Reported EPS', np.nan)
        if pd.isna(est) or pd.isna(act) or est == 0:
            continue
        surprise = act - est
        surprise_pct = surprise / est

        future = hist.index[hist.index > date]
        past = hist.index[hist.index < date]
        if future.empty or past.empty:
            continue
        next_day = future[0]
        prev_day = past[-1]
        price_prev = hist.loc[prev_day, 'Close']
        price_next = hist.loc[next_day, 'Close']
        ret_next = (price_next - price_prev) / price_prev
        label = 1 if ret_next > 0 else 0

        rows.append({
            'earnings_date': date,
            'estimate': est,
            'actual': act,
            'surprise': surprise,
            'surprise_pct': surprise_pct,
            'next_day_return': ret_next,
            'label': label
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('earnings_date').reset_index(drop=True)
    return df


def main():
    start_date = dt.datetime.today() - dt.timedelta(days=15*365)
    df = get_aapl_features(start_date)
    if df.empty:
        print("No valid feature rows to train on.")
        return

    total = len(df)
    beats = (df['surprise'] > 0).sum()
    beat_pct = beats / total
    ups = (df['label'] == 1).sum()
    up_pct = ups / total

    print("Historical summary over last 15 years:")
    print(f" - Earnings beat: {beat_pct:.2%} ({beats}/{total})")
    print(f" - Earnings miss: {100-beat_pct*100:.2f}% ({total-beats}/{total})")
    print(f" - Next-day up: {up_pct:.2%} ({ups}/{total})")
    print(f" - Next-day down: {100-up_pct*100:.2f}% ({total-ups}/{total})")

    disp = df.copy()
    disp['surprise_pct'] = disp['surprise_pct'].map("{:.2%}".format)
    disp['next_day_return'] = disp['next_day_return'].map("{:+.2%}".format)
    print("\nAll earnings events (last 15 years):")
    print(disp[['earnings_date','estimate','actual','surprise_pct','next_day_return','label']])

    features = ['surprise', 'surprise_pct']
    X = df[features]
    y = df['label']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    move_acc = np.mean(cv_scores)
    print(f"\n5-fold CV movement accuracy: {move_acc:.2%}")

    next_beat = 'beat earnings' if beat_pct > 0.5 else 'underperform earnings'
    next_move = 'up' if up_pct > 0.5 else 'down'
    print("\nPrediction for next (unreported) earnings event:")
    print(f" - EPS will likely {next_beat} (historical accuracy: {beat_pct:.2%})")
    print(f" - Stock will likely go {next_move} (model CV accuracy: {move_acc:.2%})")

if __name__ == '__main__':
    main()
