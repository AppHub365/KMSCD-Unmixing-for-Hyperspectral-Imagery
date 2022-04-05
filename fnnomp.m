function x = fnnomp(A, y, maxAtom, tol)
% Non-Negative Orthogonal Matching Pursuit with a real value dictionary

r = y;
[m,n] = size(A);
x = zeros(n,1);
k = 1;
mag = 1;
s = [];
bpr = A'*r;
Q = zeros(m,maxAtom);
R = zeros(maxAtom);
Rm1 = [];
xs = [];
r_pre = zeros(size(r));

while k <= maxAtom && mag>0 && abs(norm(r_pre,2) - norm(r,2)) >= tol
    done = 0;
    zc = 0;
    Inc = [];
    l = 1;   
        bpr(s) = 0;       
        [amp,In] = sort(bpr,'descend');
    while ~done
        if amp(l) > 0       
            sint = [s;In(l)];
            anew = A(:,sint(end));
        
            qP=Q(:,1:k-1)'*anew;
            q=anew-Q(:,1:k-1)*(qP);
            nq=norm(q);
            q=q/nq;
            zin = q'*y;
            
            %%% Positivity Guarantee
        
            v = qP;
            mu = nq;
            gamma = -Rm1*v/mu;
            
            xsp = xs;  
            if ~isempty(gamma)
            if ~isempty(find(gamma<0, 1))    
                vt = abs((xsp./gamma).*(gamma<0));                
                zt = min(vt(vt>0));
            else
                zt = inf;
            end
            else
                zt = inf;
            end            
            if (zin <= zt) 
                if (zin <= zc)                                                                   
                    s = [s;In(l)];
                    anew = A(:,In(l));       
                    qP=Q(:,1:k-1)'*anew;
                    q=anew-Q(:,1:k-1)*(qP);
                    nq=norm(q);
                    q=q/nq;
                else
                    s = [s;In(l)];
                end
                done = 1;
            else
                if zc>=zin
                    s = [s;In(lc)];
                    anew = A(:,In(lc));       
                    qP=Q(:,1:k-1)'*anew;
                    q=anew-Q(:,1:k-1)*(qP);
                    nq=norm(q);
                    q=q/nq;
                    done = 1;
                elseif zt > zc,                                 
                    zc = zt;
                    lc = l;
                    l = l+1;
                else
                    l = l+1;
                end                                     
            end                                                        
        else
            done = 1;
            mag = amp(l);
        end
        if done && (mag > 0)
%              R(1:k-1,k)=qP; % Updatin R
%              R(k,k)=nq; % Updatin R
             Q(:,k)=q;
             Rm1 = [Rm1,gamma;zeros(1,size(Rm1,2)),1/mu];
             z(k)=q'*y;
             r_pre = r;
             r = r- q*z(k);
             xs = Rm1*z(1:k)';
        end
    end    
    bpr = A'*r;            
    k = k+1;
end
xs(xs < 0) = 0;
x(s) = xs;
end
