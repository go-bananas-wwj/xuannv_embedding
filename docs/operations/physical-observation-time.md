# Physical observation time and analysis bins

`data_process.observation_time` validates explicit half-open day, month or year
intervals. Author-assigned analysis bins remain separate from physical acquisition
time. `month_support` returns one row per actual observation over intersecting,
consecutive public calendar months. Unknown within-year timing gives one weighted
yearly observation; it does not manufacture twelve aerial acquisitions. Pixel
coverage is separate and must still mask unavailable public/local observations.

The current three-region preparation has 6/5/1 real local optical views. The final
Haidian Planet acquisition is April 30, while its previously authorized analysis
bin is May. Both facts are retained. Public RGB controls use their actual selected
monthly interval, rather than inheriting the private acquisition day. Harbin now
includes the authorized April image omitted from the old annual composite. The
original arrays and all earlier experimental results are retained. This is input
preparation, not evidence that the goal has been met.

Validation: 4 targeted tests and 1085 full-suite tests passed. Formal encoder
training and downstream scoring are separate subsequent gates.
