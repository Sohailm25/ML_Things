
titanic <- read.csv('titanic_project.csv')


print(head(titanic))

print(summary(titanic))

print(str(titanic))

install.packages('gmodels')
library(gmodels)
CrossTable(titanic$sex, titanic$age)

d <- density(titanic$age, na.rm = TRUE)

plot(d, main="Kernel Density Plot for Age", xlab = "Age")
polygon(d, col="wheat", border="slategrey")


boxplot(titanic$age, col = "slategrey", horizontal=TRUE, xlab="Age",
        main="Age of Titanic Passengers")

counts <- table(titanic$pclass)
barplot(counts, xlab="Passenger Class", ylab="Frequency",
        col=c("seagreen","wheat","sienna3"))

library(vcd)
mosaic(table(titanic[,c(2,4)]), shade=TRUE, legend=TRUE)









