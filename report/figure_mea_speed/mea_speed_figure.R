#~ 
#~ import pickle
#~ dicts = [pickle.load(open("/tmp/data_benchmark_{0}.pickle".format(i))) for i in range(1, 9)]
#~ 
#~ 
#~ def one_row(d):
#~     for n,t in zip(d["n_eq"], d["dt"]):
#~         print d["git_tag"],n,t
#~         
#~ for a in dicts:
#~     for b in a:
#~         one_row(b)
#~ 
#~ exit(1)
#~ 

df <- read.table("mea_speed_data.tsv")
colnames(df) <- c("tag","max_order","dt")
df$max_order <- as.numeric(as.factor(df$max_order))
df$tag <- as.factor(df$tag)

df <- subset(df, max_order > 1 | tag == "matlab" | tag == "no_simplify")

pdf("mea_speed.pdf", w=9,h=6)
plot(dt ~ max_order, df, col=as.numeric(tag),pch=as.numeric(tag) + 15, ylab="Runtime, log(seconds)", xlab="Maximal moment order", cex=1)

print(summary(lm(dt ~ max_order * tag, df)))

tgs = unique(df$tag)
new_names <- character(0)
for (i in 1:length(tgs)){
	t <- tgs[i]
	subdf <- subset(df, tag==t)
	
	linmod <- lm(dt ~ max_order ,subdf)
	abline(linmod,col=as.numeric(t),lwd=2)
	new_names[i] <- sprintf("%s) %s", letters[i], as.character(t))
} 

legend('bottomright', legend = new_names, lty=1, col=as.numeric(tgs), cex=1, lwd=2,pch=as.numeric(tgs) + 15, title="Optimisation")

dev.off()
