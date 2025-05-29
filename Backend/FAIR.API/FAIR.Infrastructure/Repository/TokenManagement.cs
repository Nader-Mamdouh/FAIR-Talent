using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Interfaces;
using FAIR.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Net;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace FAIR.Infrastructure.Repository
{
    public class TokenManagement(AppDbContext context , IConfiguration config) : ITokenManagement
    {
        public async Task<int> AddRefreshToken(string userId, string refreshToken)
        {
            context.RefreshToken.Add(new RefreshToken
            {
                UserId = userId,
                Token = refreshToken
            });
            return await context.SaveChangesAsync();
        }
        public  string GenerateToken(AppUser user)
        {
            var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(config["Jwt:Key"]!));
            var cred = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);
            var expairation = DateTime.UtcNow.AddDays(2);
            var roleName = user is Player?"Player" : "Coach";
            List<Claim> claim =[ 
               new Claim("FullName" , user.FullName!),
                new Claim(ClaimTypes.NameIdentifier,user.Id.ToString()),
                new Claim(ClaimTypes.Email,user.Email!),
                new Claim(ClaimTypes.Role , roleName!)
            ];
            var token = new JwtSecurityToken(
                issuer: config["Jwt:Issuer"],
                audience: config["Jwt:Audience"],
                claims: claim,
                expires: expairation,
                signingCredentials: cred
                );
            return new JwtSecurityTokenHandler().WriteToken(token);
        }
        public string GetRefreshToken()
        {
            const int byteSize = 64;
            byte[] randomBytes = new byte[byteSize];
            using (RandomNumberGenerator rng = RandomNumberGenerator.Create())
            {
                rng.GetBytes(randomBytes);
            }
            var token = Convert.ToBase64String(randomBytes);
            return WebUtility.UrlEncode(token);
        }
        public List<Claim> GetUserClaimsFromToken(string token)
        {
            var tokenHandler = new JwtSecurityTokenHandler();
            var jwtToken = tokenHandler.ReadJwtToken(token);
            if (jwtToken != null)
            {
                return jwtToken.Claims.ToList();
            }
            return [];
        }
        public async Task<int> UpdateRefreshToken(string refreshToken)
        {
            var user = await context.RefreshToken.FirstOrDefaultAsync(x => x.Token == refreshToken);
            if (user == null)
                return -1;
            user.Token = refreshToken;
            return await context.SaveChangesAsync();
        }
        public async Task<string> GetUserIdByRefreshToken(string refreshToken)
        {
            return (await context.RefreshToken.FirstOrDefaultAsync(x => x.Token == refreshToken))!.UserId;
        }
        public async Task<bool> ValidateRefreshToken(string refreshToken)
        {
            var user = await context.RefreshToken.FirstOrDefaultAsync(x => x.Token == refreshToken);
            return user != null;
        }

    }
}
