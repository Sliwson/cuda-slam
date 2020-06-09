R""(
    #version 330 core
    out vec4 FragColor;

    in vec3 Normal;  
    in vec3 FragPos;  
  
    uniform vec3 lightDirection;

    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform vec3 objectColor;

    void main()
    {
        // ambient
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * lightColor;
  	
        // diffuse 
        float diffuseStrength = 0.5;
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(-lightDirection);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diffuseStrength * diff * lightColor;
    
        // specular
        float specularStrength = 0.4;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 2);
        vec3 specular = specularStrength * spec * lightColor;  
        
        vec3 result = (ambient + diffuse + specular) * objectColor;
        FragColor = vec4(result, 1.0);
    }
)""
