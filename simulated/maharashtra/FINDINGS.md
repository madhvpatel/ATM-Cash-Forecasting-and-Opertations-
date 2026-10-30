# What the Maharashtra run showed

The forecast to keep is each ATM's own average times its day-of-month index. Weekday, payday, weather, and yesterday's miss do not earn a place. A holiday multiplier is visible on holiday dates and unstable across windows, so it stays off the load.

## The eight sites

| ATM | City | Site | Mean daily cash out | Day-to-day CV |
| --- | --- | --- | --- | --- |
| 7 | Nashik | Metro concourse | ₹10.7 lakh | 0.32 |
| 1 | Mumbai | Railway concourse | ₹9.1 lakh | 0.28 |
| 2 | Mumbai | Airport | ₹7.9 lakh | 0.32 |
| 3 | Pune | City-centre mall | ₹7.8 lakh | 0.33 |
| 6 | Nagpur | Tourist zone | ₹7.4 lakh | 0.30 |
| 5 | Nagpur | Fuel plaza | ₹4.9 lakh | 0.27 |
| 8 | Nashik | Medical college hospital | ₹3.1 lakh | 0.29 |
| 4 | Pune | Education complex | ₹2.7 lakh | 0.44 |

There is one ATM per site type, so location type and the ATM level are the same thing. The level spans about four to one.

## What still moves demand after that level

Factors were fit only on days before each test window. The last 90 days are the holdout. The 90 days before that are the check.

One-day WAPE on the holdout:

| Model | Last 90 days | Prior 90 days |
| --- | --- | --- |
| ATM average | 23.9% | 23.3% |
| ATM average × weekday | 23.9% | 23.3% |
| ATM average × day of month | 21.4% | 20.1% |
| Day of month × month | 21.6% | 20.1% |
| Plus Sunday and 2nd/4th Saturday | 21.6% | 20.1% |
| Plus regional holiday and the day before | 21.5% | 20.4% |
| Full exogenous stack | 21.4% | 20.4% |
| Stack plus yesterday's residual | 25.2% | 25.5% |

Seven-day sum WAPE, which is the indent horizon:

| Model | Last 90 days | Prior 90 days |
| --- | --- | --- |
| ATM average | 12.9% | 13.1% |
| Day of month | 11.3% | 9.3% |
| Day of month × month | 11.0% | 9.4% |
| Holiday add-on | 10.8% | 10.1% |
| Copy of the previous 7 days | 17.6% | 16.8% |

Read as one minus WAPE, the day-of-month model is about 79% accurate on a single day and 89–91% accurate on a week. That week number sits inside the 80–85% band on the slide. The day number does not.

Day of month is the only exogenous gain that shows up in both windows. Month helps the last window by a few tenths of a point and hurts the earlier window by the same amount. Two passes through each month are not enough to trust a month factor. Weekday is flat. Sunday and the bank-closure Saturdays do not move the error.

The holiday residual is real and small. Against a day-of-month and month baseline fit before the holdout, a regional holiday day has a median ratio of 1.08 across the whole history (344 ATM-days, 45 dates). The day before is 1.05. Putting that lift in as a multiplier over-forecast the earlier window (bias about +3%) and made the weekly error worse, 9.3% to 10.1%. The last window contains only three holiday dates, which is why the correlation there looks larger (0.19) than on the training days (0.10). Leave the flag in the panel. Do not let it change the load until a third year agrees.

Payday, after day of month is in the model, has a median ratio of 0.97. It is the same cycle counted twice.

Rain versus the leftover residual is 0.00 on the training days and 0.09 on the holdout. Temperature is −0.03 and 0.00. By city, on the training days, Mumbai rain is +0.10 and Pune, Nagpur, and Nashik are about −0.05. The sign flips, so weather is not a Maharashtra feature. The holdout has no day at or above 35°C, which is why that flag is blank there.

Yesterday's residual on top of the calendar made the one-day error worse in both windows, 21% to 25%. Once the day of month is removed, yesterday is noise.

## Indent sketch

Policy used here, and labeled as policy: a 7-day cycle, load = forecast / 0.70, no cassette cap. Ninety-six cycles in each window (eight ATMs × twelve weeks).

On the last 90 days the day-of-month load realized a CER of 0.72, with empty cycles on 2.1% of cycles and a shortfall of about ₹1.0 lakh when one happened. The holiday stack realized 0.70 with no empty cycle in that window, and 0.68 with no empty cycle in the earlier window, because it loaded too much. Copying the previous week realized 0.73 and was short on 3.1% of cycles, by about ₹5.8 lakh when short.

A 70% CER target is a wide buffer relative to a weekly error of about 10%. The calendar load meets it. Tightening toward 90% CER would be a different policy, and this run did not test it.

## What this run does not settle

Denomination mix, hopper capacity, failed withdrawals, downtime, and the CIT route are still outside the panel. February's 15-minute files can rebuild the note mix and the refill jumps for these eight machines. That is a second run, not an exogenous correlation.
