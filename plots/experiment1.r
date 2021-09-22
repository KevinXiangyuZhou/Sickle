library(dplyr)
library(ggplot2)
library(jsonlite)

data <- fromJSON('./new_running_result.json', flatten = TRUE)

data <- as.data.frame(data)

dat <- data.frame(x = rnorm(100), y = rnorm(100))
number_ticks <- function(n) {function(limits) pretty(limits, n)}

process_data <- function(data) {
  new_data <- data
  colnames(new_data)<-gsub("data.","",colnames(new_data))
  colnames(new_data)<-sub("analysis_type","type",colnames(new_data))
  # filter out three baseline and count the program solved over time
  # data$with_pruning_or_not[data$with_pruning_or_not == 0] <- "enumeration"
  # data$with_pruning_or_not[data$with_pruning_or_not == 1] <- "with analysis"
  new_data$type[new_data$with_pruning_or_not == 0] <- "Type Abstraction"
  new_data$type[new_data$with_pruning_or_not == 1 & new_data$type == "trace"] <- "Provenance Abstraction (Ours)"
  new_data$type[new_data$with_pruning_or_not == 1 & new_data$type == "value"] <- "Value Abstraction"
  return(new_data)
}
# process data: combine with pruning or not with pruning type as type
data <- process_data(data)
data2 <- data %>% filter(id <= 50 & type =="Provenance Abstraction (Ours)" & timeout == 0)
data1 <- data %>% filter(type =="Provenance Abstraction (Ours)" & timeout == 0
                         & id != 106 & id != 118 & id != 102 & id != 120) 
data1 <- data1 %>%
  mutate(mean_size=mean(user_example_cells), output_size=mean(output_cells))
data2 <- data2 %>% filter(time < 10)
# # 1, 3, 5, 10, >10
# datalt1 <- data1 %>% filter(num_consistent <= 1)
# datalt3 <- data1 %>% filter(num_consistent <= 3)
# datalt5 <- data1 %>% filter(num_consistent <= 5)
# datalt10 <- data1 %>% filter(num_consistent <= 10)
# datagt10 <- data1 %>% filter(num_consistent > 10)
# print(nrow(datalt1))


# data_filtered <- data %>% filter(data$num_program <= 2)
# data_filtered <- data %>% filter(data$num_program > 2 | id == 42 | id == 49)
# print(nrow(data_filtered))

data_with_a <- data %>% 
  filter(type =="Provenance Abstraction (Ours)") %>% 
  arrange(time) %>% 
  mutate(solved = row_number() - 1)
data_with_a$solved[data_with_a$timeout == 1] <- 65

# data_with_a <- data_with_a %>%  mutate(solved = round(solved / nrow(data_with_a), 2) * 100)

# process time exp data
data_with_va <- data %>% 
  filter(type =="Value Abstraction") %>%
  arrange(time) %>% 
  mutate(solved = row_number() - 1)
data_with_va$solved[data_with_va$timeout == 1] <- 48

data_without_v <- data %>% 
  filter(type =="Type Abstraction") %>% 
  arrange(time) %>% 
  mutate(solved = row_number() - 1)
data_without_v$solved[data_without_v$timeout == 1] <- 43

processed_data <- data_with_a

# plot time exp data
plot_time_1 <- processed_data %>% ggplot() +
  geom_step(aes(color = type, 
                x = time, y = solved, 
                group = interaction(type, with_pruning_or_not)))  +
  scale_x_continuous(limits = c(0, 601)) +
  scale_y_continuous(limits = c(0, 70)) +
  scale_y_continuous(breaks=number_ticks(6)) +
  scale_x_continuous(breaks = c(10, 60, 120, 300, 600), labels = c("10", "60", "120", "300", "600")) +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  labs(color = "Abstraction Type", x = "Time (seconds)", y = "# of solved benchmarks") +
  geom_hline(yintercept=70, linetype=2, 
               color = "dark gray") +
  geom_hline(yintercept=36, linetype=2, 
             color = "dark gray") +
  theme(text = element_text(size=12),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.77, 0.16),
        legend.background = element_rect(fill="white",
                                         size=0.1, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")
  
print(plot_time_1)