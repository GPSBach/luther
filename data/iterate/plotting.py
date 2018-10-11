# """
# Plotting
# """

### pair plot ###

## plot the pairplot
#initial_pairs = sns.pairplot(model,diag_kind='kde')

# ### heatmap ###

# #correlation matrix
# corr = model.corr()

# # #initialize figure
# # fig, ax = plt.subplots(1,1, figsize = (1, 5), dpi=300)

# # plot the heatmap
# sns.heatmap(corr, annot = True,
#         xticklabels=corr.columns,
#         yticklabels=corr.columns,
#        cmap="cividis")

# #### histogram w/ 2 plots #####
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.hist(y,bins=50)
# ax1.set_title('price - linear')
# ax2.hist(np.log10(y),bins=50)
# ax2.set_title('price - log')
# ax2.set_xlim(3,7)

#plt.hist(X[:,3])