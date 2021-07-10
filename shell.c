#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>
#define MAXCNTARGS 100
#define size 4

int preprocessing(char ***arrayofwords);
void print(char **arrayofwords);
void clear(char ***arrayofwords);
int check(char **arrayofwords, int len);
void indexes(char **arrayofwords, int len, int *ind1, int *ind2, int *ind3);
void conveyor(char **arrayofwords, int end, int fd_r, int fd_w, int backgrnd_md);

int pid = 0;

int main(int argc, char **argv){
	char **arrayofwords = NULL;
	int cnt_arg, fd_r, fd_w, count = 0;
	int backgrnd_md = 0, ind1, ind2, ind3, end;
	while(1){
		count++;
		ind1 = ind2 = ind3 = -1;		
		fd_r = fd_w = -2;
		backgrnd_md = 0;
		fprintf(stdout, "> ");
		cnt_arg = preprocessing(&arrayofwords);	
		if (check(arrayofwords, cnt_arg)){
			fprintf(stderr, "Incorrect command\n");
		}
		else{
			backgrnd_md = !strcmp(arrayofwords[cnt_arg-1], "&");
			indexes(arrayofwords, cnt_arg, &ind1, &ind2, &ind3); /*< >,>>*/	
			if (ind2 != -1 && ind3 != -1){
				printf("ind'<'=%d ind'>'=%d ind'>>'=%d\n", ind1, ind2, ind3);
				fprintf(stderr, "Incorrect command\n");
			}
			else{
			/********/	
				/*printf("mode = %d\n", backgrnd_md);
				printf("ind'<'=%d ind'>'=%d ind'>>'=%d\n", ind1, ind2, ind3);*/
				if (ind1 != -1)
					if ((fd_r = open(arrayofwords[ind1+1], O_RDONLY)) == -1){
						fprintf(stderr, "Wrong file %s\n", arrayofwords[ind1+1]);
						clear(&arrayofwords);
						continue;
					} 
				if (ind2 != -1)
					if ((fd_w = open(arrayofwords[ind2+1], O_WRONLY|O_TRUNC|O_CREAT, 0666)) == -1){
						fprintf(stderr, "Wrong file %s\n", arrayofwords[ind2+1]);
						close(fd_r);
						clear(&arrayofwords);
						continue;
					}
				if (ind3 != -1)
					if ((fd_w = open(arrayofwords[ind3+1], O_WRONLY|O_APPEND|O_CREAT, 0666)) == -1){
						fprintf(stderr, "Wrong file %s\n", arrayofwords[ind3+1]);
						close(fd_r);
						clear(&arrayofwords);
						continue;
					}
				if (ind1 == -1 && ind2 == -1 && ind3 == -1)
					end = cnt_arg;
				else{
					if (ind1 != -1)
						end = ind1;
					if (ind2 != -1)
						if (end != -1)
							end = end < ind2 ? end : ind2;
					if (ind3 != -1)		
						if (end != -1)
							end = end < ind3 ? end : ind3;
				}
				if (backgrnd_md)
					if (end == cnt_arg)
						end--;
				conveyor(arrayofwords, end, fd_r, fd_w, backgrnd_md);
				
				
			}
			/*********/	
		}	
		clear(&arrayofwords);
	}
	return 0;
}

int preprocessing(char ***arrayofwords){
	int c = 0, i = 0, j, ind, count = 1;
	c = getchar();
	*arrayofwords = malloc(sizeof(char *) * MAXCNTARGS);
	for (j = 0; j < MAXCNTARGS; j++)
		(*arrayofwords)[j] = NULL;
	while (c != EOF && c != '\n' && i < MAXCNTARGS - 3){
		j = 0;	
		ind = 0;
		while (c == ' ')
			c = getchar();
		while(c != EOF && c != '\n' && c != '|' && c != '>' && c != '<' && c != '&' && c != ' '){
			if (j == 0){
				(*arrayofwords)[i] = realloc((*arrayofwords)[i], sizeof(char)*size*count);
				count++;
				j = size;
			}
			(*arrayofwords)[i][ind] = c;
			ind++;
			j--;
			c = getchar();
		}			
		if (c == ' '){
			if (ind){
				if (j == 0)
					(*arrayofwords)[i] = realloc((*arrayofwords)[i], sizeof(char)*size*count);
				(*arrayofwords)[i][ind] = '\0';
			}
			while (c == ' ')
				c = getchar();
		}
		else if (c == '|' || c == '<' || c == '&'){
			if (ind){
				if (j == 0)
					(*arrayofwords)[i] = realloc((*arrayofwords)[i], sizeof(char)*size*count);
				(*arrayofwords)[i][ind] = '\0';
			}
			else{
				i--;
			}
			(*arrayofwords)[++i] = malloc(sizeof(char)*size);
			(*arrayofwords)[i][0] = c;
			(*arrayofwords)[i][1] = '\0';
			c = getchar();
		}
		else if (c == '>'){
			if (ind){
				if (j == 0)
					(*arrayofwords)[i] = realloc((*arrayofwords)[i], sizeof(char)*size*count);
				(*arrayofwords)[i][ind] = '\0';
			}
			else{
				i--;
			}
			(*arrayofwords)[++i] = malloc(sizeof(char)*size);
			c = getchar();
			if (c == '>'){
				(*arrayofwords)[i][0] = '>';
				(*arrayofwords)[i][1] = '>';
				(*arrayofwords)[i][2] = '\0';
				c = getchar();
			}
			else{
				(*arrayofwords)[i][0] = '>';
				(*arrayofwords)[i][1] = '\0';
			}
		}
		else{
			if (j == 0)
				(*arrayofwords)[i] = realloc((*arrayofwords)[i], sizeof(char)*size*count);
			(*arrayofwords)[i][ind] = '\0';
		}
		i++;
	}
	return i;
}

int check(char **arrayofwords, int len){
	int i, c = 0, length;
	int pred = -1, current = -1;
	int cnt1 = 0, cnt2 = 0, cnt3 = 0; /*< >,>> &*/
	c = arrayofwords[0][0];
	length = strlen(arrayofwords[0]);
	current = (c=='|' || c=='<' || c=='>' || c=='&') && (length == 1);
	current = current || !strcmp(arrayofwords[0], ">>");
	if (current)
		switch (c){
			case '<':
				cnt1++;
				break;
			case '>':
				cnt2++;
				break;
			case '&':
				cnt3++;
				break;
			default:
				break;
		}
	for (i = 1; i < len; i++){
		pred = current;
		c = arrayofwords[i][0];
		length = strlen(arrayofwords[i]);
		current = (c=='|' || c=='<' || c=='>' || c=='&') && (length == 1);
		current = current || !strcmp(arrayofwords[i], ">>");
		if (current)
			switch (c){
				case '<':
					cnt1++;
					break;
				case '>':
					cnt2++;
					break;
				case '&':
					cnt3++;
					break;
				default:
					break;
			}
		if ((pred + current) > 1)
			return 1;
	}
	if ((pred == -1) && current)
		return 1;
	if (cnt1 > 1 || cnt2 > 1 || cnt3 > 1)
		return 1;
	return 0;
}

void indexes(char **arrayofwords, int len, int *ind1, int *ind2, int *ind3){
	int i, c, lenw;
	/*< >,>>*/	
	for (i = 0; i < len; i++){
		c = arrayofwords[i][0];
		switch(c){
			case '<':{
				if (*ind1 < 0)
					*ind1 = i;
				break;
			}
			case '>':{
				if ((lenw = strlen(arrayofwords[i])) == 1){
					if (*ind2 < 0)
						*ind2 = i;
				}
				else if (lenw == 2 && arrayofwords[i][1] == '>'){
					if (*ind3 < 0)
						*ind3 = i;
				}
				break;
			}	
		}
	}
	return;
}

void handler(int s){
	waitpid(pid, NULL, WNOHANG);
	return;
}

void conveyor(char **arrayofwords, int end, int fd_r, int fd_w, int backgrnd_md){
	int fd1[2], fd2[2];
	int i, j, start = 0, cnt = 0, bias, pid1;
	signal(SIGCHLD, handler);
	if (backgrnd_md){
		if (fd_r == -2)
			if((fd_r = open("/dev/null", O_RDONLY)) == -1){
				fprintf(stderr, "Error: dev/null\n");
				return;
			}
		if (fd_w == -2)
			if((fd_w = open("/dev/null", O_WRONLY)) == -1){
				fprintf(stderr, "Error: dev/null\n");
				return;
			}
		
		if (!(pid = fork())){
			signal(SIGINT, SIG_IGN);
			conveyor(arrayofwords, end, fd_r, fd_w, 0);
			fprintf(stderr, "\nThe background process %d completed!\n", getpid());
			fprintf(stdout, "> ");
			clear(&arrayofwords);
			exit(0);
		}
		else{
			if (!(pid1 = fork()))
				return;
			else{
				signal(SIGINT, SIG_IGN);
				waitpid(pid1, NULL, 0);
				/*fprintf(stderr, "\nFather process %d completed!\n", getpid());*/
				clear(&arrayofwords);
				exit(0);
			}
				
		}
	}
	for (i = 0; i < end; i++){
		if (!strcmp(arrayofwords[i], "|"))
			cnt++;
	}
	/**************/
	/*printf("cnt = %d\n", cnt);
	printf("ind end = %d\n", end);*/
	pipe(fd1);
	pipe(fd2);
	for (i = 0; i < cnt+1; i++){
		
		for (j = start; j < end && strcmp(arrayofwords[j], "|"); j++);
		/*printf("j = %d\n", j);*/
		bias = start;
		start = j + 1;
		if (i>1){
			if (i % 2 == 0){
				/*printf("pipe1\n");*/
				pipe(fd1);
			}
			else{
				/*printf("pipe2\n");*/
				pipe(fd2);
			}
		}
		if (!(fork())){
			/*son*/
			arrayofwords[j] = NULL;
			/*print(arrayofwords + bias);*/
			if (!i){
				if (fd_r != -2){
					dup2(fd_r, 0);
					close(fd_r);
				}
			}
			else{
				if (i % 2 == 1){
					/*printf("i = %d\n", i);
					printf("input = %d\n", fd1[0]);*/
					dup2(fd1[0], 0);
				}
				else{
					/*printf("i = %d\n", i);
					printf("input = %d\n", fd2[0]);*/
					dup2(fd2[0], 0);
				}
			}	
			if (i == cnt){
				if (fd_w != -2){
					dup2(fd_w, 1);
					close(fd_w);
				}
			}
			else{
				if (i % 2 == 0){
					/*printf("i = %d\n", i);
					printf("ouput = %d\n", fd1[1]);*/
					dup2(fd1[1], 1);
				}
				else{
					/*printf("i = %d\n", i);
					printf("ouput = %d\n", fd2[1]);*/
					dup2(fd2[1], 1);
				}
			}
			close(fd1[0]);
			close(fd2[0]);
			close(fd1[1]);
			close(fd2[1]);
			execvp(arrayofwords[bias], arrayofwords + bias);
			fprintf(stderr, "command '%s' not found, terminate\n", arrayofwords[bias]);
			exit(-1);
		}
		if (i>0){
			if (i % 2 == 0){
				close(fd2[0]);
				close(fd2[1]);
			}
			else{
				close(fd1[0]);
				close(fd1[1]);
			}
		}
		/*father*/
		
	}
	/*printf("I = %d\n", i);*/
	if (i > 0){
		if (i % 2 == 0){
			if (i == cnt){
				close(fd1[0]);
				close(fd1[1]);
				close(fd2[0]);
				close(fd2[1]);
			}
			else{
				close(fd2[0]);
				close(fd2[1]);
			}
		}
		else{
			if (i == cnt){
				close(fd1[0]);
				close(fd1[1]);
				close(fd2[0]);
				close(fd2[1]);
			}
			else{
				close(fd1[0]);
				close(fd1[1]);
			}
		}
	}
	while(wait(NULL) != -1);
	return;
}



void print(char **arrayofwords){
	int i = 0;
	while (arrayofwords[i]){
		printf("%s\n", arrayofwords[i]);
		i++;
	}
}

void clear(char ***arrayofwords){
	int i = 0;
	while ((*arrayofwords)[i]){
		free((*arrayofwords)[i]);
		i++;
	}
	free(*arrayofwords);
	*arrayofwords = NULL;
	return;
}
