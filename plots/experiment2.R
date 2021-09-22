library(dplyr)
library(ggplot2)
library(jsonlite)
library(ggpubr)

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

# data <- data %>% mutate(md = median(input_cells))

# data <- data  %>% filter(timeout == 0) 

# data <- data %>%
#   group_by(id) %>% 
#   mutate(cnt = n()) %>% 
#   filter(cnt == 3)

# data <- data %>% group_by(type) %>%
#   summarise(mean_n = mean(num_program_visited), mean_time = mean(time))




data_filtered_e <- data %>% filter(data$num_program <= 2 & id != 42 & id != 49)
data_filtered_h <- data %>% filter(data$num_program > 2 | id == 42 | id == 49)

data_filtered_h <- data_filtered_h %>%
  filter(timeout == 0) %>% 
  group_by(id) %>% 
  mutate(cnt = n()) %>% 
  filter(cnt == 3)
data_filtered_h <- data_filtered_h %>% group_by(type) %>%
  summarise(mean_n = mean(num_program_visited), mean_time = mean(time))

print(nrow(data_filtered_e))
print(nrow(data_filtered_h))
# 34 hard cases
# 36 easy cases

data_with_a1 <- data_filtered_e %>% 
  filter(type =="Provenance Abstraction (Ours)") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_a2 <- data_filtered_h %>% 
  filter(type =="Provenance Abstraction (Ours)") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_a2$solved[data_with_a2$timeout == 1] <- 30

# data_with_a <- data_with_a %>%  mutate(solved = round(solved / nrow(data_with_a), 2) * 100)

# process time exp data
data_with_va1 <- data_filtered_e %>% 
  filter(type =="Value Abstraction") %>%
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_va2 <- data_filtered_h %>% 
  filter(type =="Value Abstraction") %>%
  arrange(time) %>% 
  mutate(solved = row_number())
data_with_va2$solved[data_with_va2$timeout == 1] <- 14

data_without_v1 <- data_filtered_e %>% 
  filter(type =="Type Abstraction") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_without_v2 <- data_filtered_h %>% 
  filter(type =="Type Abstraction") %>% 
  arrange(time) %>% 
  mutate(solved = row_number())
data_without_v2$solved[data_without_v2$timeout == 1] <- 8

processed_data1 <- rbind(data_with_a1, data_with_va1, data_without_v1)
processed_data2 <- rbind(data_with_a2, data_with_va2, data_without_v2)
# ----------------------time exp----------------------
plot_time_1 <- processed_data1 %>% ggplot() +
  geom_step(aes(color = type, 
                x = time, y = solved, 
                group = interaction(type, with_pruning_or_not)))  +
  scale_x_continuous(limits = c(0, 30)) +
  scale_y_continuous(limits = c(-2, 40), expand=c(0,0)) +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  labs(color = "Abstraction Type", x = "Time (seconds)", y = "# of solved benchmarks") +
  geom_hline(yintercept=36, linetype=2, 
             color = "dark gray") +
  scale_y_continuous(breaks=number_ticks(6)) +
  scale_x_continuous(breaks = c(1, 2, 4, 8, 24), labels = c("1", "2", "4", "8", "24")) +
  theme(text = element_text(size=11),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.81, 0.14),
        legend.background = element_rect(fill="white",
                                         size=0.09, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

dat <- data.frame(x = rnorm(100), y = rnorm(100))
number_ticks <- function(n) {function(limits) pretty(limits, n)}

plot_time_2 <- processed_data2 %>% ggplot() +
  geom_step(aes(color = type,
                x = time, y = solved, 
                group = interaction(type, with_pruning_or_not)))  +
  scale_x_continuous(limits = c(0, 601)) +
  scale_y_continuous(limits = c(-2, 35), expand=c(0,0)) +
  labs(color = "Abstraction Type", x = "Time (seconds)", y = "# of solved benchmarks") +
  geom_hline(yintercept=34, linetype=2, 
             color = "dark gray") +
  scale_y_continuous(breaks=number_ticks(6)) +
  scale_x_continuous(breaks = c(10, 60, 120, 300, 600), labels = c("10", "60", "120", "300", "600")) +
  theme(text = element_text(size=11),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.81, 0.14),
        legend.background = element_rect(fill="white",
                                         size=0.09, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")


# ----------------------plot num searched exp----------------------
# data_level_3 <- data %>% filter(data$num_program > 3)
# data_level_12 <- data %>% filter(data$num_program <= 3 & data$time < 10 )
plot_num_searched <- data_filtered_h %>% ggplot(aes(x=type,
                                     y=num_program_visited, 
                                     fill=type)) + 
  geom_boxplot(width=0.5) + 
  geom_jitter(width=0.3, size = 1.3) +
  # scale_color_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) +
  labs(color = "Abstraction Type", x = "", y = "# of solved benchmarks") +
  scale_y_continuous(breaks=number_ticks(6)) +
  theme(text = element_text(size=9),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.17, 0.87),
        legend.background = element_rect(fill="white",
                                         size=0.05, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

plot_num_searched_1 <- data_filtered_e %>% ggplot(aes(x=type,
                                                 y=num_program_visited, 
                                                 fill=type)) + 
  geom_boxplot(width=0.5) + 
  geom_jitter(width=0.3, size = 1.3) +
  labs(color = "Abstraction Type", x = "", y = "# of solved benchmarks") +
  coord_cartesian(ylim = c(0, 1000)) +
  scale_y_continuous(breaks=number_ticks(6)) +
  theme(text = element_text(size=9),
        legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_blank(),
        panel.background = element_blank(),
        panel.border = element_rect(linetype = "solid", color="gray10", fill = NA),
        legend.position = c(0.17, 0.87),
        legend.background = element_rect(fill="white",
                                         size=0.05, linetype="solid", 
                                         colour ="gray"),
        axis.line = element_line(colour = "black"),
        strip.background=element_rect(fill="white"),
        strip.placement = "outside")

# print(plot_random)
# print(plot_size_exp)
print(plot_num_searched)
print(plot_num_searched_1)
print(plot_time_1)
print(plot_time_2)

exp2 <- ggarrange(plot_time_1, plot_time_2, plot_num_searched_1, plot_num_searched, 
          ncol = 4, nrow = 1)
# print(exp2)



